import smtplib
import ssl


class Notification:

    def __init__(self):
        # self.port = 465
        # self.smtp_server_domain_name = "smtp.gmail.com"
        # self.sender_mail = "etllumon@gmail.com"
        # self.password = "Javeriana2021&"
        self.sender_mail = 'gabriel.moreno@lumon.com.co'
        self.port = 465
        self.smtp_server_domain_name = "in-v3.mailjet.com"
        self.user = "c463f1bf579cf08229add3f8bcf67e6d"
        self.password = "a5ed5ab13d8790f6f09130efd032b114"

    def send(self, emails, subject, content):
        ssl_context = ssl.create_default_context()
        service = smtplib.SMTP_SSL(self.smtp_server_domain_name, self.port, context=ssl_context)
        service.login(self.user, self.password)

        for email in emails:
            result = service.sendmail(self.sender_mail, email, f"Subject: {subject}\n{content}")

        service.quit()


if __name__ == '__main__':
    # mails = input("Enter emails: ").split()
    # subject = input("Enter subject: ")
    # content = input("Enter content: ")
    # https://app.mailjet.com/account

    mails = ["morenoluis@javeriana.edu.co", "gabrielmoreno10@gmail.com"]
    subject = "test email"
    content = "Other example for email..."

    mail = Notification()
    mail.send(mails, subject, content)

