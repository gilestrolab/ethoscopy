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
import re, fnmatch
import json
import datetime
import hashlib

DB_FOLDER = '/opt/ethoscope_metadata'


def md5file(filename):
    with open(filename, "rb") as f:
        file_hash = hashlib.md5()
        while chunk := f.read(8192):
            file_hash.update(chunk)
            
    return file_hash.hexdigest()


class db_organiser():
    
    _csv_filename = 'ethoscope_db.csv'
    db_list = []
    
    def __init__(self, db_path, refresh=True, fullpath=False, csv_path=DB_FOLDER):
        self.db_path = db_path
        self._csv_path = csv_path
        
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
        db_list = self._update_db_list()
        

        #this takes 17 seconds
        # df = pd.DataFrame(columns=['ethoscope_id', 'ethoscope_name', 'experiment_date', 'experiment_time', 'db_filename'])
        # for db in db_list:
            # e_id, e_name, exp_datetime, db_name = db.split("/")
            # exp_date, exp_time = exp_datetime.split("_")
            # df.loc[len(df)] = [e_id, e_name, exp_date, exp_time, db_name]
     
        #while this takes 0.2 seconds
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
        self.db.to_csv( os.path.join(self._csv_path, self._csv_filename),
                        index=True )
        
    def loaddb(self):
        '''
        Load a previously saved db description
        '''
        self.db = pd.read_csv( os.path.join(self._csv_path, self._csv_filename),
                               index_col=[0] )
        
        
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
      
    
    def __init__(self, filename : str, project='unnamed', tags=[], authors = [], doi = '', description='', separator='auto'):
        '''
        '''
    
        self.db_folder = os.path.join (DB_FOLDER, 'db')     
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
            self.info = { 'project' : self._sanitise(project),
                          'tags' : tags,
                          'description': description,
                          'authors' : authors,
                          'paper_doi' : doi
                         }
        
        self._create_summary()


    def _sanitise(self, project):
        '''
        Make sure the name is appropriate as a path
        '''
        
        prj = str(project).strip().replace(' ', '_')
        return re.sub(r'(?u)[^-\w.]', '', prj )
        

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
        
        metadata_hash = md5file(self.filename)
        
        info = {
            'filename' : self.filename,
            'entries' : self.data.shape[0],
            'columns' : list(self.data.columns),
            'db_files' : self.db_files.shape[0],
            'info_mtime' : f"{datetime.datetime.now():%Y-%m-%d %H:%M}",
            'metadata_hash' : metadata_hash,
            'identifier' : '%s-%s' % (self.info['project'] , metadata_hash )
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

        self.info.update(info)
        return self.info

    def list_dbs(self, notfound = False):
        '''
        Return a list of all the db files associated with this metadata
        '''
        
        if notfound:
            return self.db_files.loc[self.db_files.db_filename.isna()]
        else:
            return self.db_files.loc[self.db_files.db_filename.notna()]
        

    def save( self, project = None):
        '''
        Save metadata file locally
        ''' 

        if project is not None and project != self.info['project']:
            # we are changing project name
            old_project_name = self.info['project']
            self.info['project'] = self._sanitise( project )

        # get filepath
        _, filename = os.path.split(self.filename)
        full_dir = os.path.join( self.db_folder, self.info['project'] )
        full_path = os.path.join( full_dir, filename)

        # create project dir
        try:
            os.mkdir(full_dir)
        except:
            pass
            
        # overwrite metadata
        self.data.to_csv(full_path, index=False)

        # save new json summary
        self.filename = full_path
        self.info['filename'] = full_path
        
        json_file = full_path + ".info"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.summary, f, ensure_ascii=False, indent=4)


        return True
        

class metadata_crawler():

        
    def __init__(self):
        self.db_folder = os.path.join (DB_FOLDER, 'db')
        self.upload_folder = os.path.join(self.db_folder, 'unnamed')

        self.refresh_all_info()


    def refresh_all_info(self):
        '''
        Gather all the info from the info files and stores them in one unique dictionary
        '''
        #Refresh the list of info files
        self.crawl()

        self.all_info = {}
        self.all_projects = {}

        for info_file in self.all_info_files:
            with open(info_file, 'r') as i:
                j = json.load(i)
                prj_name = j['project']
                identifier = '%s-%s' % ( prj_name , j['metadata_hash'])
                self.all_info.update ( {identifier : j })
                
                if prj_name not in self.all_projects:
                    self.all_projects[ prj_name ] = {}

                self.all_projects[ prj_name ].update ({ os.path.split (j['filename'])[1] : identifier})

    @property
    def available_options(self):
        '''
        Crawl the info and collects the choice of authors, DOIs, and projects
        Useful for autocompletion choices
        '''
        
        def remove_empty(l):
            
            #flatten the list
            fl = lambda l:[element for item in l for element in fl(item)] if type(l) is list else [l]
            #remove empty items
            return list(set([i for i in fl(l) if ((i != '') and (i != [])) ]))
        
        available_options = { 'project' : [] , 
                              'authors' : [] , 
                              'paper_doi' : [],
                              'tags' : [],
                              'metadata_hash' : [] 
                              }

        for info in self.all_info:
            for key in available_options:
                try:
                    available_options[key].append( self.all_info[info][key] )
                except:
                    # this file is missing this key
                    pass
        
        for key in available_options:
            try:
                available_options[key] = remove_empty( available_options[key] )
            except:
                # this key is missing
                pass

        return available_options
        
    
    def crawl(self):
        '''
        returns a list of all the CSV files found in the db folder
        '''
        self.all_info_files = []
        self.info_tree = {}
        
        for root, dirnames, filenames in os.walk(self.db_folder):
            for filename in fnmatch.filter(filenames, '*.info'):
                self.all_info_files.append( os.path.join(root, filename) )

                dirname = os.path.basename(root)
                if dirname not in self.info_tree: self.info_tree.update ({dirname : {}})
                
                self.info_tree[ dirname ].update ({ filename: hash })
    
    def request(self, identifier):
        '''
        return the json content of a specific info file
        '''

        if identifier in self.all_info:
            return self.all_info[identifier]

        else:
            self.refresh_all_info()
            try:
                return self.all_info[identifier]
            except:
                return {}
        
    def find(self, criteria):
        '''
        example of criteria
        { 'authors' : 'Gilestro',
          'project' : '*2022*'
        }
        These will be treated as AND gate
        '''
        all_matches = {}
        match = {}
        
        for key in criteria:
            for info in self.all_info:
                #try:
                if type(self.all_info[info][key]) is list: 
                    tobesearched = self.all_info[info][key]
                else:
                    tobesearched = [self.all_info[info][key]]
                    
                if fnmatch.filter( tobesearched, criteria[key]):
                    try:
                        all_matches[info] += 1
                    except: 
                        all_matches[info] = 1
                
                #except KeyError as e:
                #    pass
                    #raise e # key not found 
                    
        for identifier in all_matches:
            if all_matches[identifier] == len(criteria):
                match.update ({ identifier : self.all_info[identifier] })
            
        
        return match
                
             

if __name__ == '__main__':

    #datapath = "/mnt/data/results"
    #metadata_filename = '/home/gg/Downloads/METADATA.txt'

    #db = db_organiser(datapath, refresh=False)
    mdb = metadata_crawler()
    for hash_id in mdb.all_info:
        info = mdb.all_info[hash_id]
        meta = metadata_handler(info['filename'])
        print (meta.summary)
    
    #meta.associate_to_db(db)
    #print (meta.data)
