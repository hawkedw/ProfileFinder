@echo off
setlocal

echo === ProfileFinder build ===

pushd "%~dp0"

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "PROJECT_ROOT=%%~fI"

if exist "%PROJECT_ROOT%\.venv\Scripts\python.exe" (
    set "PYTHON=%PROJECT_ROOT%\.venv\Scripts\python.exe"
) else if exist ".venv\Scripts\python.exe" (
    set "PYTHON=.venv\Scripts\python.exe"
) else (
    set "PYTHON=python"
)

set "BUILD_WORKDIR=build\pyinstaller"

"%PYTHON%" -m pip install -r requirements.txt
if errorlevel 1 goto :error

"%PYTHON%" -m PyInstaller --noconfirm --clean --distpath "%PROJECT_ROOT%" --workpath "%BUILD_WORKDIR%" ProfileFinder.spec
if errorlevel 1 goto :error

echo.
echo Done. Executable: %PROJECT_ROOT%\ProfileFinder.exe
popd
pause
exit /b 0

:error
echo.
echo Build failed.
popd
pause
exit /b 1
