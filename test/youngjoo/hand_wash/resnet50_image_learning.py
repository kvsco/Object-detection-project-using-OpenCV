
from tensorflow.keras.applications import ResNet50
from keras.layers import Dense, Input, Activation
from keras.models import Model


from keras.layers import BatchNormalization,Dropout,Flatten,MaxPooling2D
from matplotlib import pyplot as plt

input = Input(shape=(150, 150, 3))
model = ResNet50(input_tensor=input, include_top=False, weights="imagenet")

x = model.output
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Flatten()(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu', input_dim=input)(x)
x = BatchNormalization()(x)
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dense(5, activation='softmax')(x)

new_model = Model(inputs=model.input, outputs=x)
# CNN Pre-trained 가중치를 그대로 사용할때
for layer in new_model.layers[:19] :
    layer.trainable = False

new_model.summary()

new_model.compile(loss='sparse_categorical_crossentropy',
                     optimizer='adam',
                     metrics=['accuracy'])



from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 위노그라드 알고리즘 설정
train_dir ='./dataset/training_set'
test_dir = './dataset/test_set'

# 폴더에 따라 자동 분류
train_image_generator = ImageDataGenerator(
    rotation_range=40,       #회전범위
    width_shift_range=0.3,   #수평 랜덤평행이동 0.2%
    height_shift_range=0.3,  #수직 랜덤평행이동 0.2%
    rescale=1./255,          #0-1범위로 스케일링
    shear_range=0.2,         #임의 전단변환 (?)
    zoom_range=0.3,          #임의 확대축소변환
    fill_mode = 'nearest')
test_image_generator = ImageDataGenerator(
    rotation_range=40,       #회전범위
    width_shift_range=0.3,   #수평 랜덤평행이동 0.2%
    height_shift_range=0.3,  #수직 랜덤평행이동 0.2%
    rescale=1./255,          #0-1범위로 스케일링
    shear_range=0.2,         #임의 전단변환 (?)
    zoom_range=0.3,          #임의 확대축소변환
    fill_mode = 'nearest')

# 데이터 구조 생성
train_data_gen = train_image_generator.flow_from_directory(batch_size=32,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(150, 150),
                                                           class_mode='binary')

test_data_gen = test_image_generator.flow_from_directory(batch_size=32,
                                                         directory=test_dir,
                                                         target_size=(150, 150),
                                                         class_mode='binary')

history = new_model.fit(train_data_gen, epochs=40,
                        validation_data=test_data_gen)

new_model.save("resnet50_wash_hand.h5")

# 최종 결과 리포트
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))



plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='testing acc')
plt.title('Training and testing accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='testing loss')
plt.title('Training and testing loss')
plt.legend()

plt.show()