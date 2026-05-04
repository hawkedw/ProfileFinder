# ProfileFinder

Repository layout is split into two zones:

- Root contains the current distributable build: `ProfileFinder.exe`
- Root also contains `input/`, where DSM and CSV input files can be placed before running the app
- `distr/` contains the source code and build files used to assemble `ProfileFinder.exe`

## What To Give The Customer

For a simple handoff, use the root-level files:

- `ProfileFinder.exe`
- `README.md`
- `input/`

The application lets the user pick files interactively and save results where needed. If an `output/` folder does not exist yet, it will be created automatically when saving results.

## Source And Build Files

Everything needed for development and rebuilding the executable lives under `distr/`:

- `distr/app.py`
- `distr/core.py`
- `distr/line_profile_locator.py`
- `distr/profilePoints/`
- `distr/requirements.txt`
- `distr/ProfileFinder.spec`
- `distr/build.bat`

## Rebuild The Executable

Run:

```bat
distr\build.bat
```

The rebuilt executable will be written to:

```text
ProfileFinder.exe
```
