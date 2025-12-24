@REM ----------------------------------------------------------------------------
@REM File:       docs/make.bat
@REM Project:    Baby LLM - Unified Neural Child Development System
@REM Created by: Celaya Solutions, 2025
@REM Author:     Christopher Celaya <chris@chriscelaya.com>
@REM Description: Batch file for building documentation on Windows
@REM Version:    1.0.0
@REM License:    MIT
@REM Last Update: January 2025
@REM ----------------------------------------------------------------------------

@ECHO OFF

pushd %~dp0

REM Command file for Sphinx documentation

set SPHINXBUILD=sphinx-build
set SOURCEDIR=.
set BUILDDIR=_build

if "%1" == "" goto help

%SPHINXBUILD% >NUL 2>NUL
if errorlevel 9009 (
	echo.
	echo.The 'sphinx-build' command was not found. Make sure you have Sphinx
	echo.installed, then set the SPHINXBUILD environment variable to point
	echo.to the full path of the 'sphinx-build' executable. Alternatively you
	echo.may add the Sphinx directory to PATH.
	echo.
	echo.If you don't have Sphinx installed, grab it from
	echo.https://sphinx-doc.org/
	exit /b 1
)

%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
goto end

:help
%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%

:end
popd
