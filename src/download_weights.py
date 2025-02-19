from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os


def create_drive_service(creds_path="mycreds.txt"):
    """
    Создаёт объект GoogleDrive, используя локальную аутентификацию,
    с сохранением/загрузкой учетных данных в creds_path.
    """
    gauth = GoogleAuth()

    # Пытаемся загрузить существующие учетные данные
    if os.path.exists(creds_path):
        gauth.LoadCredentialsFile(creds_path)

    # Если их нет или они просрочены — запускаем процесс авторизации
    if not gauth.credentials or gauth.credentials.invalid:
        # Локальный веб-сервер откроет браузер для входа в Google-аккаунт
        gauth.LocalWebserverAuth()
        # Можно сохранить полученные учётные данные в файл
        gauth.SaveCredentialsFile(creds_path)

    return GoogleDrive(gauth)


def download_files(save_folder="downloads"):
    ids = {
        'stylegan': '1EM87UquaoQmk17Q8d5kYIAHqu0dkYqdT',
        'encoder': '1cUv_reLE6k3604or78EranS7XzuVMWeO',
    }

    drive = create_drive_service()

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for name, file_id in ids.items():
        downloaded = drive.CreateFile({'id': file_id})
        downloaded.FetchMetadata(fetch_all=True)
        file_path = os.path.join(save_folder, downloaded.metadata['title'])
        downloaded.GetContentFile(file_path)
