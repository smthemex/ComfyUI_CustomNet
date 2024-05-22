# A CustomNet node for ComfyUI   
A CustomNet node for ComfyUI   



NOTICE
----
整合包的模块引用问题已经解决，目前下载即用，如果报错，尝试安装需求文件里的库。


CustomNet  From: [link](https://github.com/TencentARC/CustomNet)
----

1.Installation
-----
  In the .\ComfyUI \ custom_node directory, run the following:   
  
  ``` python 
  git clone https://github.com/smthemex/ComfyUI_CustomNet.git     
  ```

  
2.requirements  
----
每个人的环境不同，但是carvekit-colab是必须装的，是内置的脱底工具包，有空我再去掉这个，用外置的脱底节点。首次运行，会安装carvekit-colab的模型文件，无梯子的注意。    
need carvekit-colab==4.1.0    

3 Download the model 
----
下载customnet_v1.pth模型，并放在pretrain目录下  
Download the weights of Customnet customnet_v1.pth and put it to ./pretrain   [link](https://huggingface.co/TencentARC/CustomNet/tree/main)   

首次使用会下载3个的模型文件，分别是      
clip：文件目录一般在C:/User/你的用户名/.cache/clip/ViT-L-14.pt  
carvekit的2个脱底模型：  
目录C:/User/你的用户名/.cache/carvekit/checkpoints/fba/fba_matting.pth     
目录C:/User/你的用户名/.cache/carvekit/checkpoints/tracer_b7/tracer_b7.pth   


4 Other
----
目前只能测试玩玩，尺寸及详细调参要有空才更新了。    

5 Example
-----
normal  常规脱底置于提示测的背景前面         
![](https://github.com/smthemex/ComfyUI_CustomNet/blob/main/example/example.png)

polar   主体上下视角     
![](https://github.com/smthemex/ComfyUI_CustomNet/blob/main/example/polar.png)

zaimuth   主体左右视角     
![](https://github.com/smthemex/ComfyUI_CustomNet/blob/main/example/zaimuth.png)

position X0 Y0  主体在背景中的位置    
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
