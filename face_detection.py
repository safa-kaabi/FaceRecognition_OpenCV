from PIL import Image, ImageDraw
import numpy as np
import face_recognition


safa_image=face_recognition.load_image_file("./Khnow/safa.png")
safa_face_encoding = face_recognition.face_encodings(safa_image)[0]

yoser_image=face_recognition.load_image_file("./Khnow/yosser.jpg")
yosser_face_encoding = face_recognition.face_encodings(yoser_image)[0]

cristiano_image=face_recognition.load_image_file("./Khnow/cristiano.jfif")
cristiano_face_encoding = face_recognition.face_encodings(cristiano_image)[0]

messi_image=face_recognition.load_image_file("./Khnow/messi.jfif")
messi_face_encoding = face_recognition.face_encodings(messi_image)[0]

Mbappe_image=face_recognition.load_image_file("./Khnow/Mbappe.jfif")
Mbappe_face_encoding = face_recognition.face_encodings(Mbappe_image)[0]

known_face_encodings = [cristiano_face_encoding,messi_face_encoding,safa_face_encoding,Mbappe_face_encoding,yosser_face_encoding]

known_face_names = ["Cristiano Rolando","Messi","Safa Kaabi","Kylian Mbappe" ,"Yosser Kaabi"]

image = input("please enter image number : ")
unknown_image = face_recognition.load_image_file(f'./UnKnown/{image}.png')


face_locations =face_recognition.face_locations(unknown_image) 
Unknown_face_encodings=face_recognition.face_encodings(unknown_image,face_locations)

pil_image=Image.fromarray(unknown_image)
draw = ImageDraw.Draw(pil_image )

for (top, right, bottom, left), Unknown_face_encodings in zip(face_locations, Unknown_face_encodings):

   matches= face_recognition.compare_faces(known_face_encodings,Unknown_face_encodings)
   name="unknown"

face_distance = face_recognition.face_distance(known_face_encodings,Unknown_face_encodings)
print(face_distance)

best_match_index = np.argmin(face_distance)
print(best_match_index)
print(matches)

if matches[best_match_index]:
        name = known_face_names[best_match_index]

draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

text_width, text_height = draw.textsize(name)
draw.rectangle(((left, bottom - text_height - 10),
                   (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
draw.text((left + 6, bottom - text_height - 5),
              name, fill=(255, 255, 255, 255))

del draw

pil_image.show()