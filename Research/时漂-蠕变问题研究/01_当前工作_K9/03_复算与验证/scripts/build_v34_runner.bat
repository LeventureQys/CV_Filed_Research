@echo off
rem Build the plan-v3.4 offline diagnostic runner (links PRODUCT source only).
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
set ROOT=D:\workshop\Processing\multi-device-cascade-host-cpp
set S=%ROOT%\temp\v4.1flash\plan\v3.4\scripts
cd /d "%S%\build" 2>nul || (mkdir "%S%\build" && cd /d "%S%\build")
cl /nologo /EHsc /utf-8 /std:c++17 /O2 /I "%ROOT%\src" /I "%ROOT%\thirdparty\eigen-5.0.0" /Fe:v34_runner.exe "%S%\v30_runner.cpp" "%ROOT%\src\domain\drift_v6\drift_v6_compensator.cpp"
if errorlevel 1 exit /b 1
echo BUILD OK
