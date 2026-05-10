@echo off
title NFN AGI — Neural Fractal Network
echo.
echo  ^+══════════════════════════════════════^+
echo  ^|   Neural Fractal Network             ^|
echo  ^|   AGI ^— All-in-One App               ^|
echo  ^|                                      ^|
echo  ^|   Chat ^· Train ^· Explore ^· Adapt    ^|
echo  ^+══════════════════════════════════════^+
echo.

where python >nul 2>&1
if errorlevel 1 (
    echo  [ERREUR] Python introuvable sur votre systeme.
    echo.
    echo  Installez Python 3.10 ou superieur depuis :
    echo  https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

python run.py %*
if errorlevel 1 (
    echo.
    echo  [ERREUR] Le serveur NFN AGI a rencontre une erreur.
    echo.
    echo  Conseils de depannage :
    echo    1. Verifiez que Python 3.10+ est installe
    echo    2. Installez les dependances : pip install -r requirements.txt
    echo    3. Consultez les logs ci-dessus pour plus de details
    echo.
    pause
)
