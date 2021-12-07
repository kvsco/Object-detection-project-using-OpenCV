## hand_joint
 + openCV 영상데이터에서 손가락 관절(joint) 추출
 + 코드 설명 주석참조

## image_extract 
 + openCV로 영상 불러와서 프레임단위로 쪼개 이미지파일로 저장하는 코드
 + local에서 영상 폴더경로 작성의 유의, 10초짜리 영상이 300장 넘게 extract 댐

## model_test 
 + 학습된 모델 테스트하는 코드.
 + load_model에 학습된 모델 가져오고 경로설정

## vggNet_image_learn
 + 실행환경 jupyter notebook
 + 마지막 dense층 분류 갯수 유의, 2개이상 softmax()
 + image_generator 적당히 파라미터값 조절 (주석에 자세한 설명)
