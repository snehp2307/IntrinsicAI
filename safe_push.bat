rmdir /s /q .git
git clone https://github.com/snehp2307/IntrinsicAI.git temp_clone
xcopy temp_clone\.git .git\ /E /H /C /I /Y
rmdir /s /q temp_clone
git add .
git commit -m "Refactored valuation module, NIFTY 500 support, yfinance fallback"
git push origin main
