def define_generator():
  train_generator = ImageDataGenerator(
    rotation_range = 80,
    zoom_range = 0.2,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    shear_range = 0.2,
    vertical_flip = True,
    horizontal_flip = True,
    
  )
  validation_generator = ImageDataGenerator(
    rotation_range = 80,
    zoom_range = 0.2,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    shear_range = 0.2,
    vertical_flip = True,
    horizontal_flip = True,
    
  )
  test_generator = ImageDataGenerator(
    rotation_range = 80,
    zoom_range = 0.2,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    shear_range = 0.2,
    vertical_flip = True,
    horizontal_flip = True,
    
  )

  train_generator = train_generator.flow_from_directory(
        directory = PATH_TRAIN,
        target_size = (IMAGE_SIZE, IMAGE_SIZE),
        batch_size = BATCH_SIZE,
        color_mode = 'rgb',
        class_mode = "sparse",
  )
  validation_generator = validation_generator.flow_from_directory(
        directory = PATH_VAL,
        target_size = (IMAGE_SIZE, IMAGE_SIZE),
        batch_size = BATCH_SIZE,
        color_mode = 'rgb',
        class_mode = "sparse",
  )
  test_generator = test_generator.flow_from_directory(
        directory = PATH_TEST,
        target_size = (IMAGE_SIZE, IMAGE_SIZE),
        batch_size = BATCH_SIZE,
        color_mode = 'rgb',
        class_mode = "sparse",
  )
  return train_generator, validation_generator , test_generator
