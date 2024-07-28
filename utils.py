import base64


def decodeString(image_b64, imagePath):
    # Decode the base64 string
    image_data = base64.b64decode(image_b64)

    # Write the decoded data to a file
    with open(imagePath, 'wb') as image_file:
        image_file.write(image_data)
