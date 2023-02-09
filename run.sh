# sisr: real
python run_sisr.py test -s saved/sisr/real/x4/biqrnn -r best
python run_sisr.py test -s saved/sisr/real/x4/mcnet -r best
python run_sisr.py test -s saved/sisr/real/x4/sspsr -r best

python run_sisr.py test -s saved/sisr/real/x8/biqrnn -r best
python run_sisr.py test -s saved/sisr/real/x8/mcnet -r best
python run_sisr.py test -s saved/sisr/real/x8/sspsr -r best

python run_sisr.py test -s saved/sisr/real/x16/biqrnn -r best
python run_sisr.py test -s saved/sisr/real/x16/mcnet -r best
python run_sisr.py test -s saved/sisr/real/x16/sspsr -r best

# sisr: flower
python run_sisr.py test -s saved/sisr/flower/x4/biqrnn -r best
python run_sisr.py test -s saved/sisr/flower/x4/mcnet -r best
python run_sisr.py test -s saved/sisr/flower/x4/sspsr -r best

python run_sisr.py test -s saved/sisr/flower/x8/biqrnn -r best
python run_sisr.py test -s saved/sisr/flower/x8/mcnet -r best
python run_sisr.py test -s saved/sisr/flower/x8/sspsr -r best

python run_sisr.py test -s saved/sisr/flower/x16/biqrnn -r best
python run_sisr.py test -s saved/sisr/flower/x16/mcnet -r best
python run_sisr.py test -s saved/sisr/flower/x16/sspsr -r best

# refsr: flower
python run_refsr.py test -s saved/refsr/flower/sf4 -r best
python run_refsr.py test -s saved/refsr/flower/sf8 -r best
python run_refsr.py test -s saved/refsr/flower/sf16 -r best
python run_refsr.py test -s saved/refsr/flower/ablation/sisr -r best 
python run_refsr.py test -s saved/refsr/flower/ablation/simple4 -r best 
python run_refsr.py test -s saved/refsr/flower/ablation/simple8 -r best 
python run_refsr.py test -s saved/refsr/flower/ablation/wo-atten -r best 

# refsr: real
python run_refsr.py test -s saved/refsr/real/sf4 -r best
python run_refsr.py test -s saved/refsr/real/sf8 -r best
python run_refsr.py test -s saved/refsr/real/sf16 -r best