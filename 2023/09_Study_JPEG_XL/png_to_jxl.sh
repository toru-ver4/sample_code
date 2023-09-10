#!/bin/sh
cjxl ./img/D65_202_HLG.png ./img/D65_202_HLG.jxl -q 100 -x color_space=RGB_D65_202_Rel_HLG
cjxl ./img/D65_202_PeQ.png ./img/D65_202_PeQ.jxl -q 100 -x color_space=RGB_D65_202_Rel_PeQ
cjxl ./img/D65_DCI_PeQ.png ./img/D65_DCI_PeQ.jxl -q 100 -x color_space=RGB_D65_DCI_Rel_PeQ
cjxl ./img/D65_SRG_SRG.png ./img/D65_SRG_SRG.jxl -q 100 -x color_space=RGB_D65_SRG_Rel_SRG
cjxl ./img/D65_SRG_709.png ./img/D65_SRG_709.jxl -q 100 -x color_space=RGB_D65_SRG_Rel_709
