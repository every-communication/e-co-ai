    
![지문자_이미지](https://user-images.githubusercontent.com/90700892/209419165-fd373820-e70c-4a1b-b2a4-c82439db4c1c.jpg)

출처 : https://www.urimal.org/1222

### ***Pipeline***

- making_video.py
    - 원하는 자,모음을 설정해 동영상을 생성합니다. (openCV 활용)
    
- create_dataset_from_video.py
    - video data를 사용하여 hand keypoint의 Vector, Angle 값을 sequence data로 변환해 npy 파일로 저장합니다.
    
- train_hand_gesture.ipynb
    - npy file load하여 모델을 생성합니다.
    
- video_test_model_tflite.py
    - videoFolderPath를 지정하여 저장된 비디오를 활용하여 테스트합니다.
    
- webcam_test_model_tflite.py
    - webcam을 활용하여 실시간으로 테스트합니다.