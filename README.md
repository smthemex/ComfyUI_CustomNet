# A CustomNet node for ComfyUI   
A CustomNet node for ComfyUI   

CustomNet: Object Customization with Variable-Viewpoints in Text-to-Image Diffusion Models.
CustomNet  From: [CustomNet](https://github.com/TencentARC/CustomNet)

Update
----
2024/08/11
--同步官方的内绘模型及代码，优化模型加载方式，现在模型跟常规的SD模型在一个地方，优化模型加载方式，

1.Installation
-----
  In the .\ComfyUI \ custom_node directory, run the following:   
  
  ``` python 
  git clone https://github.com/smthemex/ComfyUI_CustomNet.git     
  ```
2.requirements  
----
每个人的环境不同，但是carvekit-colab是必须装的，是内置的脱底工具包，懒得去掉了，你可以先用其他sam节点处理物体图。首次运行，会安装carvekit-colab的模型文件，无梯子的注意。    
need carvekit-colab==4.1.0    

3 Download the model 
----
3.1 normal：  
下载customnet_v1.pth模型，并放在ComfyUI/models/checkpoints/目录下：  
Download the weights of Customnet “customnet_v1.pth” and put it to “ComfyUI/models/checkpoints/”   [link](https://huggingface.co/TencentARC/CustomNet/tree/main)   
```
└── ComfyUI/models/checkpoints/
    ├── customnet_v1.pth
 ```   
3.2 inpainting：   
下载customnet_inpaint_v1.pt模型，并放在ComfyUI/models/checkpoints/目录下：  
Download the weights of Customnet “customnet_inpaint_v1.pt” and put it to “ComfyUI/models/checkpoints/”   [link](https://huggingface.co/TencentARC/CustomNet/tree/main)  
```
└── ComfyUI/models/checkpoints/
    ├── customnet_inpaint_v1.pt
```
3.3 clip and carvekit:
首次使用会下载3个的模型文件，须连外网：，分别是       
clip：文件目录一般在C:/User/你的用户名/.cache/clip/ViT-L-14.pt   
carvekit的2个脱底模型：  
目录C:/User/你的用户名/.cache/carvekit/checkpoints/fba/fba_matting.pth     
目录C:/User/你的用户名/.cache/carvekit/checkpoints/tracer_b7/tracer_b7.pth   

6 Tips
----
---白底的物体图得到最好的效果；
---底模训练就是256的，所以没法做大图，除非腾讯把大图的模型放出来。  
---The object image with a white background achieves the best effect；

5 Example
-----
normal  常规脱底置于提示测的背景前面，最新的演示；  Latest Presentation        
![](https://github.com/smthemex/ComfyUI_CustomNet/blob/main/example/normal.png)

inpainting  内绘模型，最新的演示；  Latest Presentation   
![](https://github.com/smthemex/ComfyUI_CustomNet/blob/main/example/inpainting.png)

polar   主体上下视角  既往的演示，   Previous demonstrations   
![](https://github.com/smthemex/ComfyUI_CustomNet/blob/main/example/polar.png)

zaimuth   主体左右视角   既往的演示，   Previous demonstrations   
![](https://github.com/smthemex/ComfyUI_CustomNet/blob/main/example/zaimuth.png)

position X0 Y0  主体在背景中的位置  既往的演示，   Previous demonstrations   
![](https://github.com/smthemex/ComfyUI_CustomNet/blob/main/example/position.png)


6 Citation
------

``` python  
@misc{yuan2023customnet,
    title={CustomNet: Zero-shot Object Customization with Variable-Viewpoints in Text-to-Image Diffusion Models}, 
    author={Ziyang Yuan and Mingdeng Cao and Xintao Wang and Zhongang Qi and Chun Yuan and Ying Shan},
    year={2023},
    eprint={2310.19784},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
}
