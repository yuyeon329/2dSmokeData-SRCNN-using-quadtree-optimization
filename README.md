# 2dSmokeData-SRCNN-using-quadtree-optimization
# 최적화된 쿼드트리를 이용한 2차원 연기 데이터의 효율적인 슈퍼 해상도 기법
### SR(Super-Resolution)을 계산하는데 필요한 데이터를 효율적으로 분류하고, 분할하여
### 빠르게 SR 연산을 가능하게 하는 쿼드트리 기반 최적화 기법을 제안하는 프로젝트입니다.  
---  

### **제안하는 쿼드트리 최적화 기법**  
전처리 단계에 쿼드트리 최적화 기법을 추가하였습니다.  

- 입력 데이터로 사용하는 연기 데이터를 DownScaling하여 쿼드트리 연산 소요 시간을 감소시킵니다.  
- DownScaling시 Binarization을 동반하여 DownScaling 과정에서 밀도가 손실되는 문제를 피합니다.  

### **네트워크**  
- VGG 19 기반 네트워크  
- Convolution 계층을 거칠 때 데이터의 손실을 막기 위해 Residual 방식과 유사하게 이전 계층의 출력 값을 더해주며 학습합니다.  

### **제안한 방법의 효과**  
- 이전 결과 기법에 비해 **약 15~18배**정도의 속도 향상을 얻었습니다.  
  
### **참고용 Input Output 비교 이미지**  
![image](https://user-images.githubusercontent.com/73763069/172090865-bcfa1c0a-a0b6-4b3b-9386-7e56ecf4ce04.png)

---

![image](https://user-images.githubusercontent.com/73763069/172088588-80084571-1ec9-41ef-90b9-537a3d1082c4.png)
![image](https://user-images.githubusercontent.com/73763069/172088637-91f1e630-07bf-414e-902f-9058bbc11af0.png)
![image](https://user-images.githubusercontent.com/73763069/172088667-0ab22ab9-b6ff-40b2-b391-ec7b49d22c9f.png)
![image](https://user-images.githubusercontent.com/73763069/172089069-0ca60498-a2ac-46aa-959c-bedb3d2c4119.png)
![image](https://user-images.githubusercontent.com/73763069/172089092-0c2b39e5-0541-4372-bd52-21586b9e21c1.png)
![image](https://user-images.githubusercontent.com/73763069/172089119-0eeb5e64-0c5d-477a-ac23-a8f453838ba6.png)
![image](https://user-images.githubusercontent.com/73763069/172089142-1efcb6ce-ee15-4b19-b738-468f22f96cd0.png)
![image](https://user-images.githubusercontent.com/73763069/172089172-11a9e270-14ed-4fee-8ac2-358e98876cad.png)
![image](https://user-images.githubusercontent.com/73763069/172089184-38786b34-22b7-40ee-8d35-34668661032b.png)
![image](https://user-images.githubusercontent.com/73763069/172089231-eb80125a-4eca-49c2-8b31-75d0b35a000c.png)
![image](https://user-images.githubusercontent.com/73763069/172089247-d8281381-7951-4afc-ab4b-eab09d189910.png)
---
