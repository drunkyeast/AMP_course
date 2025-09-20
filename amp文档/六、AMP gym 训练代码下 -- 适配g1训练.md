  
## g1 模型更改

### urdf查看
urdf 中g1 的小腿和脚又干涉，通过在mujoco中可以看到，导致的训练失败


###  创建
g1_amp_base.py
g1_amp.py
motion_lib_g1.py
G1AMP.yaml
G1AMPPPO.yaml
tasks文件夹 __init__.py  增加
    
数据文件夹 g1_motions
模型文件夹 g1_description
 
###  开源参考

https://anishhdiwan.github.io/noise-conditioned-energy-based-annealed-rewards/
