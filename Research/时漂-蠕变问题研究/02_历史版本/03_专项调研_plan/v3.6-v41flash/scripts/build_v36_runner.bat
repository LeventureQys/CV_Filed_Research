@echo off
rem Build the plan-v3.6 candidate runner (links ONLY the v3.6 variant source; product src/ untouched).
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
set ROOT=D:\workshop\Processing\multi-device-cascade-host-cpp
set V36=%ROOT%\temp\v4.1flash\plan\v3.6-v41flash
if not exist "%V36%\build" mkdir "%V36%\build"
cd /d "%V36%\build"
cl /nologo /EHsc /utf-8 /std:c++17 /O2 /I "%V36%\src_v36" /I "%ROOT%\src" /I "%ROOT%\thirdparty\eigen-5.0.0" /Fe:v36_runner.exe "%V36%\scripts\v36_runner.cpp" "%V36%\src_v36\domain\drift_v6\drift_v6_compensator.cpp"
if errorlevel 1 exit /b 1
echo BUILD OK
