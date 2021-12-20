import requests
import json
import csv
import os

if not os.path.exists("./google_rate"):
    os.mkdir("./google_rate")

user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/95.0.4638.69 Safari/537.36 Edg/95.0.1020.44"

headers = {
    "User-Agent": user_agent,
}

ss = requests.session()


def get_comments(title):
    stopRes = ")]}'\n[null,null,null,"
    counter = 0
    pagetext = ""
    url = 'https://www.google.com.tw/maps/preview/review/listentitiesreviews?authuser=0&hl=zh-TW&gl=tw&pb=!1m2!1y3777962109657537619!2y200968063298875096!2m2!1i{}0!2i10!3e1!4m5!3b1!4b1!5b1!6b1!7b1!5m2!1swJ26Yd_2C5HTmAWYy6A4!7e81'

    res = ss.get(url.format(pagetext)).text

    pretext = ')]}\''
    text = res.replace(pretext, '')
    soup = json.loads(text)
    conlist = soup[2]

    with open('./google_rate/{}.csv'.format(title), 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f,
                                fieldnames=["restaurant", "username", "comm_time", "comment", "rate"],
                                delimiter=',')
        writer.writeheader()

    try:
        for i in conlist:
            username = str(i[0][1])
            comm_time = str(i[1])
            comment = str(i[3]).replace("\n", ' ')
            rate = str(i[4])
            try:
                with open('./google_rate/{}.csv'.format(title), 'a', encoding='utf-8', newline='') as f:
                    writer = csv.DictWriter(f,
                                            fieldnames=["restaurant", "username", "comm_time", "comment", "rate"],
                                            delimiter=',')
                    writer.writerow(
                        {"restaurant": title, "username": username, "comm_time": comm_time, "comment": comment,
                         "rate": rate})
            except:
                pass
            print("username:" + str(i[0][1]))
            print("time:" + str(i[1]))
            print("comment:" + str(i[3]))
            print("rate:" + str(i[4]))
            print("=" * 20)

    except TypeError:
        pass

    while res.startswith(stopRes) is not True:
        counter = counter + 1
        pagetext = str(counter)
        res = ss.get(url.format(pagetext), headers=headers).text
        print("-" * 20)

        pretext = ')]}\''
        text = res.replace(pretext, '')
        soup = json.loads(text)
        conlist = soup[2]
        try:
            for i in conlist:
                username = str(i[0][1])
                comm_time = str(i[1])
                comment = str(i[3]).replace("\n", ' ')
                rate = str(i[4])
                try:
                    with open('./google_rate/{}.csv'.format(title), 'a', encoding='utf-8', newline='') as f:
                        writer = csv.DictWriter(f,
                                                fieldnames=["restaurant", "username", "comm_time", "comment", "rate"],
                                                delimiter=',')
                        writer.writerow(
                            {"restaurant": title, "username": username, "comm_time": comm_time, "comment": comment,
                             "rate": rate})

                except:
                    pass
                print("username:" + str(i[0][1]))
                print("time:" + str(i[1]))
                print("comment:" + str(i[3]))
                print("rate:" + str(i[4]))
                print("=" * 20)
        except TypeError:
            pass


if __name__ == "__main__":
    get_comments(input("店家名稱: "))
