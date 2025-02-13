import smtplib # for sending email
from email.mime.text import MIMEText # for sending email
from email.mime.multipart import MIMEMultipart # for sending email
import os
from email.mime.base import MIMEBase
from email import encoders
from email.mime.image import MIMEImage

# sendEmail(accuracy, fileName, newBestAccuracy, False) # trainingCompleted boolean
def sendEmail(bestAccuracy, fileName, newBestAccuracy, loss, graphName, trainingComplete):
    # Email details
    sender_email = "williamcorkey1@gmail.com"
    receiver_email = "wfcorkey@gmail.com"
    password = "jpgi eofu dwjx gmyf"  # I use an app password through Google Account to allow Python access

    # Set up the server
    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()  # Secure the connection
    server.login(sender_email, password)

    # Create email content
    subject = "Accuracy Increase to " + str(round(bestAccuracy, 3))
    if newBestAccuracy:
        subject += ", new best accuracy"
    if trainingComplete:
        subject += ", > 85%, goodbye world o7"
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject
    message.attach(MIMEText(f"The accuracy has increased to {bestAccuracy}\nThe loss is {loss}", "plain"))


    if os.path.exists(fileName):  # Ensure the file exists
        if fileName.lower().endswith(".png"):
            # Attach as PNG image using MIMEImage
            with open(fileName, "rb") as attachment:
                img = MIMEImage(attachment.read(), _subtype="png")
                img.add_header("Content-Disposition", f"attachment; filename={os.path.basename(fileName)}")
                message.attach(img)
        else:
            # Attach as a generic file type
            with open(fileName, "rb") as attachment:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(attachment.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(fileName)}")
            message.attach(part)
    # # Attach the .txt file
    # if os.path.exists(fileName):  # Ensure the file exists
    #     with open(fileName, "rb") as attachment:
    #         part = MIMEBase("application", "octet-stream")  # Generic file type
    #         part.set_payload(attachment.read())  # Read the file content

    #     encoders.encode_base64(part)  # Encode file as base64
    #     part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(fileName)}")

    #     message.attach(part)  # Attach the file

    # Send the email
    server.sendmail(sender_email, receiver_email, message.as_string())
    server.sendmail(sender_email, "bdemissew@elon.edu", message.as_string())