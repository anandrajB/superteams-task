import os
import tempfile
from pathlib import Path
from zipfile import ZipFile


async def create_zip_from_files(files):
    zip_path = "files.zip"
    with tempfile.TemporaryDirectory() as temp_dir:
        for file in files:
            file_content = await file.read()
            file_path = Path(temp_dir) / file.filename
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "wb") as f:
                f.write(file_content)
        with ZipFile(zip_path, "w") as zip_file:
            for root, _, filenames in os.walk(temp_dir):
                for filename in filenames:
                    file_path = os.path.join(root, filename)
                    arc_name = os.path.relpath(file_path, temp_dir)
                    zip_file.write(file_path, arc_name)

    return zip_path
