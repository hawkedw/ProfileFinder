@echo off
echo === ProfileFinder build ===

pip install -r requirements.txt

python -m PyInstaller --onefile --windowed --name ProfileFinder --collect-all rasterio --collect-all pyproj --collect-all numpy app.py

echo.
echo Done. Executable: dist\ProfileFinder.exe
pause
