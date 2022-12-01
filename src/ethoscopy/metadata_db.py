#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  metadata_db.py
#  
#  Copyright 2022 Giorgio F. Gilestro <giorgio@gilest.ro>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#  
import pandas as pd
import fnmatch
import os
from math import floor, log
import re
import json
import datetime

class db_organiser():
    
    _csv_filename='ethoscope_db.csv'
    db_list = []
    
    def __init__(self, db_path, refresh=True, fullpath=False):
        self.db_path = db_path
        
        if refresh:
            self.update_dataframe()
            self.savedb()
        else:
            self.loaddb()
            
            
    def _update_db_list(self, include_root=False): 
        '''
        Returns a python list with the fullpaths of all the *.db files
        :param boolean include_root: whether the root part of the path should be included in the output
        
        The structure of each entry is:
        
        if include_root is False:
            ETHOSCOPE_ID/ETHOSCOPE_NAME/YYYY-MM-DD_HH-MM-SS/YYYY-MM-DD_HH-MM-SS_ETHOSCOPE_ID.db
        else: 
            ROOT_PATH/ETHOSCOPE_ID/ETHOSCOPE_NAME/YYYY-MM-DD_HH-MM-SS/YYYY-MM-DD_HH-MM-SS_ETHOSCOPE_ID.db
        '''
        
        all_db_files = []
        
        for root, dirnames, filenames in os.walk(self.db_path):
            for filename in fnmatch.filter(filenames, '*.db'):
                fp = os.path.join(root, filename)
                
                if not include_root:
                    all_db_files.append( os.path.relpath(fp, self.db_path ))
                else:
                    all_db_files.append( fp )
                    
        return all_db_files


    def update_dataframe(self, fullpath=True):
        '''
        '''
        #this takes 17 seconds
        # df = pd.DataFrame(columns=['ethoscope_id', 'ethoscope_name', 'experiment_date', 'experiment_time', 'db_filename'])
        # for db in db_list:
            # e_id, e_name, exp_datetime, db_name = db.split("/")
            # exp_date, exp_time = exp_datetime.split("_")
            # df.loc[len(df)] = [e_id, e_name, exp_date, exp_time, db_name]
     
        #while this takes 0.2 seconds
        
        db_list = self._update_db_list()
        
        l = []
        for db in db_list:
            #takes only the last four entries of the path, otherwise it will fail if fullpath=True 
            e_id, e_name, exp_datetime, db_name = db.split("/")[-4:]
            exp_date, exp_time = exp_datetime.split("_")
            filesize = os.stat(os.path.join(self.db_path, db)).st_size
            
            if not fullpath:
                l.append ([e_id, e_name, exp_date, exp_time, db_name, filesize])
            else:
                l.append ([e_id, e_name, exp_date, exp_time, db, filesize])
        
        self.db = pd.DataFrame(l, columns=['ethoscope_id', 'ethoscope_name', 'experiment_date', 'experiment_time', 'db_filename', 'filesize'])

    def savedb(self):
        '''
        Save a db description
        '''
        self.db.to_csv(self._csv_filename, index=True)
        
    def loaddb(self):
        '''
        Load a previously saved db description
        '''
        self.db = pd.read_csv(self._csv_filename, index_col=[0])
        
        
    def make_index_file(self, filename='index.txt'):
        '''
        Creates a rethomics index file used for FTP purposes, describing all the contents of the folder
        '''
        index_file = os.path.join(self.db_path, filename)
                
        with open(index_file, "w") as ind:
            
            for db in self._update_db_list():
                fs = os.stat(fp).st_size
                ind.write('"%s", %s\r\n' % (db, fs))


    def find_entry(self, ethoscope_name=None, experiment_date=None, experiment_time=None, db_filename=None):
        '''
        A general way to interrogate the db
        '''
        
        if db_filename:
            return self.db.loc [ (self.db['db_filename'] == db_filename) ]
            
        elif ethoscope_name and experiment_date and experiment_time:
            return self.db.loc [ (self.db['ethoscope_name'] == ethoscope_name) & (self.db['experiment_date'] == experiment_date) & (self.db['experiment_time'] == experiment_time)]

        elif ethoscope_name and experiment_date and not experiment_time:
            return self.db.loc [ (self.db['ethoscope_name'] == ethoscope_name) & (self.db['experiment_date'] == experiment_date)]

        elif ethoscope_name and not experiment_date:
            return self.db.loc [ (self.db['ethoscope_name'] == ethoscope_name) ]

        elif not ethoscope_name and experiment_date:
            return self.db.loc [ (self.db['experiment_date'] == experiment_date) ]

    def find_db_file(self, row):
        '''
        Find the appropriate db file for the given row of metadata
        When two dbs are present, always returns the db with the largest filesize
        '''
        
        g = self.find_entry( row['machine_name'], row['date'] )
        try:
            r = g.loc [ g['filesize'].idxmax() ][['db_filename', 'filesize']]
        except:
            r = [pd.NA, pd.NA]
        return r

    def add_files_to_metadata(self, metadata):
        '''
        Finds the db_filenames in the provided db that match each of the metadata instances
        '''
        metadata[['db_filename', 'filesize']] = metadata.apply(self.find_db_file, axis = 1,  result_type='expand')
        return metadata


class metadata_handler():
    
    _DBFOLDER = './db/'
    
    def __init__(self, filename, project='unnamed', tags=[], authors = [], doi = '', description='', separator='auto'):
        
        self.filename = filename
        _ , extension = os.path.splitext(filename)
        
        if separator == 'auto' and extension == '.tsv':
            separator = '\t'
        elif separator == 'auto': #for csv and any other format we default to commas
            separator = ','
        
        self.data = pd.read_csv(filename, sep=separator)

        try:
            info_file = filename+'.info'
            with open(info_file, 'r') as i:
                self.info = json.load(i)
        except:
            self.info = { 'tags' : tags,
                          'project' : project,
                          'description': description,
                          'authors' : authors,
                          'paper_doi' : doi
                         }
        
        self._create_summary()

    def associate_to_db(self, db):
        '''
        Finds the db_filenames in the provided db that match each of the metadata instances
        '''

        self.data = db.add_files_to_metadata(self.data)
        self._create_summary()
    
    def has_db_info(self):
        '''
        '''
        return 'db_filename' in self.data
    
    def _create_summary(self):
        '''
        '''
        if 'db_filename' in self.data:
            self.db_files = self.data.groupby(['date','machine_name', 'db_filename', 'filesize'], dropna=False)[['date','machine_name', 'db_filename', 'filesize']].size().to_frame('count').reset_index()
        else:
            self.db_files = self.data.groupby(['date','machine_name'])[['date','machine_name']].size().to_frame('count').reset_index()
            
    @property    
    def summary(self):
        '''
        Returns some information about the metadata
        '''

        def format_bytes(size):
          mag = 0 if size <= 0 else floor(log(size, 1024))
          return f"{round(size / 1024 ** mag, 2)} {['B', 'KB', 'MB', 'GB', 'TB'][int(mag)]}"
        
        info = {
            'filename' : self.filename,
            'entries' : self.data.shape[0],
            'columns' : list(self.data.columns),
            'db_files' : self.db_files.shape[0],
            'info_mtime' : f"{datetime.datetime.now():%Y-%m-%d %H:%M}"
            } 

        if self.has_db_info():
            info.update({
            'entries_not_found' : int(self.data.db_filename.isna().sum()),
            'db_files_na' : int(self.db_files.db_filename.isna().sum()),
            'db_files_size' : format_bytes(self.db_files.filesize.sum())
            })
        else:
            info.update({
            'entries_not_found' : 'N/A',
            'db_files_na' : 'N/A',
            'db_files_size' : 'N/A'
            })

        info.update(self.info)    
        return info

    def list_dbs(self, notfound = False):
        '''
        Return a list of all the db files associated with this metadata
        '''
        
        if notfound:
            return self.db_files.loc[self.db_files.db_filename.isna()]
        else:
            return self.db_files.loc[self.db_files.db_filename.notna()]
        

    def save( self, project='unnamed', filename=None ):
        '''
        Save metadata file locally
        ''' 

        def sanitise_path(s):
            s = str(s).strip().replace(' ', '_')
            return re.sub(r'(?u)[^-\w.]', '', s)
        #try:
        if not filename:
            filename = self.filename
        _, filename = os.path.split(filename)

        full_dir = os.path.join( self._DBFOLDER, sanitise_path(self.info['project'] ) )
        full_path = os.path.join( full_dir, filename)

        try:
            os.mkdir(full_dir)
        except:
            pass
            
        #save metadata
        self.data.to_csv(full_path, index=False)

        #save json summary
        self.filename = full_path
        self.info['filename'] = full_path
        
        json_file = full_path + ".info"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.summary, f, ensure_ascii=False, indent=4)

        return True
        #except:
        #    return False
        

class db_crawler():
    
    _DBFOLDER = './db'
        
    def __init__(self):
        self.all_info_files = self.crawl()
        self._all_info = self.populate_info()

    @property
    def all_projects(self):
        return self._all_info['projects']
    
    @property
    def all_authors(self):    
        return self._all_info['authors']

    @property
    def all_dois(self):    
        return self._all_info['dois']

    @property
    def all_tags(self):    
        return self._all_info['tags']

    @property
    def all_info(self):    
        return self._all_info


    def populate_info(self):
        '''
        Crawl the info files and collects the choice of authors, DOIs, and projects
        '''
        
        def remove_empty(l):
            return list(set([i for i in l if ((i != '') and (i != [])) ]))
        
        all_projects = []
        all_authors = []
        all_dois = []
        all_tags = []
        
        for info_file in self.all_info_files:
            with open(info_file, 'r') as i:
                j = json.load(i)
            all_projects.append(j['project'])
            all_dois.append (j['paper_doi'])
            all_tags += j['tags']
            all_authors += j['authors']

        return {'projects' : remove_empty(all_projects), 
                'authors' : remove_empty(all_authors), 
                'dois' : remove_empty(all_dois), 
                'tags' : remove_empty(all_tags)
                 }
    
    def crawl(self):
        '''
        returns a list of all the CSV files found in the db folder
        '''
        all_metadata_files = []
        
        for root, dirnames, filenames in os.walk(self._DBFOLDER):
            for filename in fnmatch.filter(filenames, '*.info'):
                all_metadata_files.append( os.path.join(root, filename) )
                    
        return all_metadata_files 
        

if __name__ == '__main__':

    datapath = "/mnt/data/results"
    metadata_filename = '/home/gg/Downloads/METADATA.txt'

    db = db_organiser(datapath, refresh=False)
    meta = metadata_handler(metadata_filename)
    
    meta.associate_to_db(db)
    print (meta.data)
