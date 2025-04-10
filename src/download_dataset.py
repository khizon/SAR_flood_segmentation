import gdown
import os
import zipfile

if __name__ == '__main__':
    # URL of your Google Drive file
    url = "https://drive.google.com/uc?id=1hIQT9htXSLpBYzY4-_iuZIzjLTNkOr6Y"
    # Destination path where the file will be saved
    output = os.path.join(os.getcwd(), 'sen1floods11.zip')

    # Download the file
    print("Downloading the file...")
    gdown.download(url, output, quiet=False)

    # Check if the file exists
    if os.path.exists(output):
        print(f"File downloaded successfully: {output}")
        # Unzip the file
        extract_path = os.path.join(os.getcwd(), 'sen1floods11')
        print(f"Unzipping the file to: {extract_path}")
        with zipfile.ZipFile(output, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print("Unzipping completed!")
    else:
        print("Download failed or file not found!")
