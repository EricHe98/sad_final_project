from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
# import sys

gauth = GoogleAuth()
gauth.LocalWebserverAuth()
drive = GoogleDrive(gauth)

file_list = drive.ListFile({'q':"'1hO0_OIbRYuxdHRhwWxrxnWQc4LPDSs1r' in parents and trashed=False"}).GetList()
print('Downloading {} files'.format(len(file_list)))

for file in file_list:
	print('Downloading {}'.format(file['title']))
	file.GetContentFile('data/raw/{}'.format(file['title']))
