@echo off
rem Build the v3.4 creep-observer offline sweep tool (links PRODUCT source only).
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
set ROOT=D:\workshop\Processing\multi-device-cascade-host-cpp
set S=%~dp0
if not exist "%S%build" mkdir "%S%build"
cd /d "%S%build"
cl /nologo /EHsc /utf-8 /std:c++17 /O2 /I "%ROOT%\src" /I "%ROOT%\thirdparty\eigen-5.0.0" ^
   /Fe:v34_sweep.exe "%S%v34_sweep.cpp" "%ROOT%\src\domain\drift_v6\creep_observer.cpp"
if errorlevel 1 exit /b 1
echo BUILD OK
