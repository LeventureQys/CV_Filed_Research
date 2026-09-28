@echo off
rem Build the plan-v3.2 candidate runner (links ONLY the v3.2 variant source; product src/ untouched).
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
set ROOT=D:\workshop\Processing\multi-device-cascade-host-cpp
set V32=%ROOT%\temp\v4.1flash\plan\v3.2
if not exist "%V32%\build" mkdir "%V32%\build"
cd /d "%V32%\build"
cl /nologo /EHsc /utf-8 /std:c++17 /O2 /I "%V32%\src_v32" /I "%ROOT%\src" /I "%ROOT%\thirdparty\eigen-5.0.0" /Fe:v32_runner.exe "%V32%\scripts\v32_runner.cpp" "%V32%\src_v32\domain\drift_v6\drift_v6_compensator.cpp"
if errorlevel 1 exit /b 1
echo BUILD OK
