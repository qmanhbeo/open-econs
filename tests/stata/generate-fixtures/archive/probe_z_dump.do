clear all
log using "C:\Users\manhn\Desktop\open-econs\tests\stata\generate-fixtures\archive\probe_z_dump.log", replace text
import delimited "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\inputs\df_panel.csv", clear
xtset entity time

// ---- construct instruments EXACTLY as xtabond2 listed ----
// diff equation moments (per period j=2,3,4):
//   diff GMM: lag1 of L.y = y_{t-2}; lag(2/4) of L.y = y_{t-3}, y_{t-4}, y_{t-5}
//   iv(diff): D.x, D.z
// level equation moments:
//   level GMM: D.L.y = y_{t-1}-y_{t-2}; DL.L.y = y_{t-2}-y_{t-3}
//   iv(level): x, z, _cons
gen yl2 = L2.y
gen yl3 = L3.y
gen yl4 = L4.y
gen yl5 = L5.y
gen Dx = D.x
gen Dz = D.z
gen DLy = D.L.y
gen D2Ly = D2.L.y

// dependent & regressors
gen Ydiff = D.y
gen Ylevel = y
gen Lyd = L.y          // regressor in diff eq (y_{t-1})
gen Lyl = L.y          // regressor in level eq (y_{t-1})

preserve
keep if time>=2 & time<=4
// columns order: yl2 yl3 yl4 yl5 Dx Dz DLy D2Ly x z _cons
gen _cons = 1
keep entity time Ydiff Ylevel Lyd Lyl Dx Dz x z yl2 yl3 yl4 yl5 DLy D2Ly _cons
order entity time Ydiff Ylevel Lyd Lyl Dx Dz x z yl2 yl3 yl4 yl5 DLy D2Ly _cons
export delimited using "C:\Users\manhn\Desktop\open-econs\tests\stata\generate-fixtures\archive\z_data_dump.csv", replace
di "rows = " _N
restore
log close
