@echo off
echo === ProfileFinder build ===

pip install -r requirements.txt

pyinstaller --onefile --windowed --name ProfileFinder app.py

echo.
echo Done. Executable: dist\ProfileFinder.exe
pause
