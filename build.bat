@echo off
echo === ProfileFinder build ===

pip install -r requirements.txt

python -m PyInstaller --onefile --windowed --name ProfileFinder ^\n  --collect-all rasterio ^\n  --collect-all pyproj ^\n  --collect-all numpy ^\n  app.py

echo.
echo Done. Executable: dist\ProfileFinder.exe
pause
