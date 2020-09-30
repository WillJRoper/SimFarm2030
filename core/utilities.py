import numpy as np
import pandas as pd
from ftplib import FTP, error_perm
import fnmatch
import os


def extract_data(path):

    #  Open the csv file
    data = pd.read_csv(path, usecols=("Lat", "Long", "Year", "Sow Month", "Ripe Time", "Yield"))

    #  Remove NaN entries
    data.dropna(subset=["Yield"], inplace=True)

    # Replace missing ripe times
    data.fillna((data["Ripe Time"].mean()), inplace=True)

    # Set up a new data frame to extract day and month from sow date with split value columns
    new = data["Sow Month"].str.split("/", n=1, expand=True)
    sow_day = new[0]
    sow_mth_ini = new[1]

    # Format months
    sow_mths = []
    for s in sow_mth_ini:
        s_str = str(s)
        if len(s_str) == 1:
            sow_mths.append("0" + s_str)
        else:
            sow_mths.append(s_str)

    # Make separate columns from new data frame
    data["Sow Day"] = np.int16(sow_day)
    data.drop(columns=["Sow Month"], inplace=True)
    data["Sow Month"] = sow_mths

    #  Extract columns into numpy arrays
    lats = data["Lat"].values
    longs = data["Long"].values
    years = data["Year"].values
    ripe_time = np.int32(data["Ripe Time"].values)
    yields = data["Yield"].values
    sow_day = data["Sow Day"].values
    sow_month = data["Sow Month"].values

    # Get extreme outliers
    lim = np.mean(yields) * 0.75
    okinds = yields > lim

    print("Removed", yields[~okinds].size, "outliers with yields < %.3f" % lim, "Tons / Hectare")

    # Eliminate extreme outliers
    lats = lats[okinds]
    longs = longs[okinds]
    years = years[okinds]
    ripe_time = ripe_time[okinds]
    yields = yields[okinds]
    sow_day = sow_day[okinds]
    sow_month = sow_month[okinds]

    print("Training on", yields.size, "Regions (data points)")

    return lats, longs, years, ripe_time, yields, sow_day, sow_month


def FTP_download(ftppath, destpath, user='abowell', passwd='SimFarm2030', only_2000=True):

    # Log in to FTP server
    try:
        ftp = FTP('ftp.ceda.ac.uk')
        ftp.login(user=user, passwd=passwd)
        print('Logged in')
    except error_perm:
        raise Exception('Log in credentials invalid')

    # Define current working directory in the FTP server
    try:
        ftp.cwd(ftppath)
    except error_perm:
        raise Exception('FTP server directory invalid')

    # Adds all files found that match critera and adds to list
    try:
        files = ftp.nlst()
    except (error_perm, resp):
        if str(resp) == '550 No files found':
            raise Exception('No files in this directory')
        else:
            raise Exception('FTP server unresponsive')

    # If only year beyond 2000 are required limit the download to these files
    if only_2000:

        # Loop over files extracting only files for years of the form 20XX
        otherfiles = []
        for name in files:
            # If file is in range include it in the files list
            if fnmatch.fnmatch(name, "*hadukgrid_uk_1km_mon_20*nc"):
                otherfiles.append(name)
    else:
        otherfiles = files

    #  Make sure destination exists
    if not os.path.exists(destpath):
        print('Destination path created at', destpath)
        os.mkdir(destpath)

    # Loops through all files in the list and downloads them to wherever specified
    for counter, item in enumerate(otherfiles):
        # Log in to FTP server once again to get current files
        ftp = FTP('ftp.ceda.ac.uk')
        ftp.login(user=user, passwd=passwd)

        # Redefine current working directory in the FTP server
        ftp.cwd(ftppath)

        # Create file to write the data out to
        localfile = open(destpath + item, 'wb')
        ftp.retrbinary('RETR ' + item, localfile.write, 1024)
        print('File ', item, 'is downloaded. \n', counter, ' out of ', len(otherfiles))
        localfile.close()


def Rolling_Median(One_D_Array, n=5):

    # Convert input array to pandas series to utilise pandas functionality
    s = pd.Series(One_D_Array)

    # Get the rolling median for a window of n size
    rolling = s.rolling(n, center=True).median()

    return rolling


if __name__ == '__main__':

    # ftp_paths = ['/badc/ukmo-hadobs/data/insitu/MOHC/HadOBS/HadUK-Grid/v1.0.0.0/1km/tas/mon/v20181126',
    #              '/badc/ukmo-hadobs/data/insitu/MOHC/HadOBS/HadUK-Grid/v1.0.0.0/1km/sun/mon/v20181126',
    #              '/badc/ukmo-hadobs/data/insitu/MOHC/HadOBS/HadUK-Grid/v1.0.0.0/1km/rainfall/mon/v20181126']

    # ftp_paths = ['/badc/ukmo-hadobs/data/insitu/MOHC/HadOBS/HadUK-Grid/v1.0.1.0/1km/tas/mon/v20190808',
    #              '/badc/ukmo-hadobs/data/insitu/MOHC/HadOBS/HadUK-Grid/v1.0.1.0/1km/rainfall/mon/v20190808']


    # ftp_paths = ['/badc/ukmo-hadobs/data/insitu/MOHC/HadOBS/HadUK-Grid/v1.0.0.0/1km/rainfall/day/v20181126',
    #              '/badc/ukmo-hadobs/data/insitu/MOHC/HadOBS/HadUK-Grid/v1.0.0.0/1km/tasmax/day/v20181126',
    #              '/badc/ukmo-hadobs/data/insitu/MOHC/HadOBS/HadUK-Grid/v1.0.0.0/1km/tasmin/day/v20181126']

    ftp_paths = ['/badc/ukmo-hadobs/data/insitu/MOHC/HadOBS/HadUK-Grid/v1.0.0.0/1km/sun/mon/v20181126']

    # ftp_paths = ['/badc/ukcp18/data/land-rcm/uk/12km/rcp85/01/tas/mon/latest',
    #              '/badc/ukcp18/data/land-rcm/uk/12km/rcp85/01/pr/mon/latest']

    # destpaths = ['/Volumes/My Passport/SimFarm_monthly/Temps/',
    #              '/Volumes/My Passport/SimFarm_monthly/Rains/']

    destpaths = ['/Volumes/My Passport/SimFarm_monthly/Suns/']

    for path, destpath in zip(ftp_paths, destpaths):
        FTP_download(path, destpath)
