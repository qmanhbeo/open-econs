*! test_basic.do - Simple test
log using "C:\Users\manhn\Desktop\open-econs\tests\stata\do\test_basic.log", replace text
di "Hello from Stata"
import delimited "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\df_panel.csv", clear
summarize y
log close
