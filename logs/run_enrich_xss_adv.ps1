export DEEPSEEK_API_KEY='sk-ffc6fa18776a475c8aba7d5457df2824'
export XSS_ENRICH_MAX_SEEDS='500'
export XSS_ENRICH_VARIANTS='4'
cd '/mnt/c/Users/HAD/Desktop/AI_in_cyber/LLM_in_Cyber'
python -u scripts/etl/enrich_xss_deepseek.py | tee logs/enrich_xss_v6_adv.log
