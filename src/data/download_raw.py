from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
# see https://cloud.google.com/docs/authentication/getting-started for 
# setting up credentials.json which must be renamed to client_secrets.json

gauth = GoogleAuth()
gauth.CommandLineAuth()
drive = GoogleDrive(gauth)

file_list = drive.ListFile({'q':"'1hO0_OIbRYuxdHRhwWxrxnWQc4LPDSs1r' in parents and trashed=False"}).GetList()
print('Downloading {} files'.format(len(file_list)))

for file in file_list:
	print('Downloading {}'.format(file['title']))
	file.GetContentFile('data/raw/{}'.format(file['title']))
