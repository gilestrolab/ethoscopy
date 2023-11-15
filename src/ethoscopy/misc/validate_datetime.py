from datetime import datetime

def validate_datetime(data):
    """ 
    Checks the date column of a pandas dataframe for the format YYYY-MM-DD, corrects formats DD-MM-YYYY
    and DD/MM/YYYY, raises an error message for other formats
    returns data unaltered if there are no alerations, if changes found the date column is updated
    @data = pandas dataframe file containing with a column headed 'date' 
    """
    date_list = data['date'].values.tolist()
    new_date_list = []
    for i, date in enumerate(date_list):
            try:
                date == datetime.strptime(date, '%Y-%m-%d').strftime('%Y-%m-%d')
            except ValueError:
                try:
                    if date == datetime.strptime(date, '%d-%m-%Y').strftime('%d-%m-%Y'):
                        date = datetime.strptime(date, '%d-%m-%Y').strftime('%Y-%m-%d')
                        new_date_list.append(date)
                except ValueError:
                    try:
                        if date == datetime.strptime(date, '%d/%m/%Y').strftime('%d/%m/%Y'):
                            date = datetime.strptime(date, '%d/%m/%Y').strftime('%Y-%m-%d')
                            new_date_list.append(date)
                    except ValueError:
                        raise ValueError("Incorrect data format, should be YYYY-MM-DD for row: " + str(i+1))

    if len(new_date_list) == 0:
        return data
    else:
        data['date'] = new_date_list   
        return data
