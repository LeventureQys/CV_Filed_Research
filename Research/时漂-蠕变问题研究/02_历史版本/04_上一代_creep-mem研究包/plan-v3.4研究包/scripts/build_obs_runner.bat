@echo off
rem Build the plan-v3.4 creep-observer offline runner (links PRODUCT source only).
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
set ROOT=D:\workshop\Processing\multi-device-cascade-host-cpp
set S=%ROOT%\temp\v4.1flash\plan\v3.4\scripts
cd /d "%S%\build"
cl /nologo /EHsc /utf-8 /std:c++17 /O2 /I "%ROOT%\src" /I "%ROOT%\thirdparty\eigen-5.0.0" /Fe:obs_runner.exe "%S%\obs_runner.cpp" "%ROOT%\src\domain\drift_v6\creep_observer.cpp"
if errorlevel 1 exit /b 1
echo BUILD OK
