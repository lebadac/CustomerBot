import re
import csv
import json  # Add this import statement for json
from MyChatbotData import MyChatbotData
from fuzzywuzzy import process
from emoji import emoji_normalize, emoji_isolate, ascii_normalize


# Initialize punct_re_escape globally
punct_re_escape = re.compile('[%s]' % re.escape('!"#$%&()*+,./:;<=>?@[\\]^_`{|}~'))


# Load training data from JSON file
def load_training_data_from_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# Load training data from CSV file
def load_training_data_from_csv(file_path):
    training_data = {}
    with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)  # Skip the header row
        for row in csv_reader:
            intent, pattern, category = row
            if intent in training_data:
                training_data[intent]['patterns'].append(pattern.strip())
            else:
                training_data[intent] = {
                    'intent': intent.strip(),
                    'patterns': [pattern.strip()],
                    'category': category.strip()
                }
    return training_data




# Paths to your files
json_file_path = './training_sample.json'
csv_file_path = './training_data.csv'

# Load training data from JSON and CSV files

json_data = load_training_data_from_json(json_file_path)
# print("Json_format:", json_data)  # Print the number of columns in CSV for verification
csv_data = load_training_data_from_csv(csv_file_path)
# print("csv_format:", csv_data)  # Print the number of rows in CSV for verification
# Kết hợp dữ liệu từ tệp JSON và tệp CSV
training_data = {**json_data, **csv_data}
# print("csv_format:", training_data) 
# Define answers dictionary

# Define answers dictionary
answers = {
    "customer_service.say_hello": {
        "text": "👋 Hello! Welcome to our customer service. I'm happy to assist you today, how can I help you?",
        "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAOEAAADhCAMAAAAJbSJIAAAByFBMVEX///+nUVQuLi793eLurr4+Gx8AAABNNznEVhnDVBXM3vKnUFQvLy/93OL/4+j/4OXqe3T1o3fwonhOODrunnPTcT6mUVVJMjSqUFMpKSnx8fGKeHj1s8QxMTEgICCtU1gAExuRkpOcUy8RAABFKi3l5eXW1ta4uLhvSTbv7u9BGyAcAABfS0w5OTnodGw+Pj5FMSY4KSNkUkoPHyN1d3eLREfQy8tmaWkfAAA8KixeXl66GjIWFhaDhoenVCo2PT8hKi2rrq6KUVJwNjmaS08kERFeLzE9ICQ0GxyDTlBWQ0SPRUlQUVFyYmMdEhEwAATaprOXd37KmqZDFhfOsrX/0NhsWFmZh4qgoKCpuMixUR5MRkVnPT9XNDWdbm6IS04vIyVySk+cWFxQKSssDhJ7UFHJnJyeQEO3d3vdxcOyZ2hhQELDi44AEQWDZGzRq6qmho20gI29pKh0VlxRHiGuk5dlKjBeUlX/xtGvdoJ3RFDckaFRKjOVYWz3qLqkaXXZlqNCAADwjaPvmJXyq6vpiYTtlZNnBhqSAA+3ABlPAACtGS6DDB4ZLCqJlqW7zN2CjZdWQjh0WU3TlXGkdFnojFueZEi9aUEe4PolAAAckUlEQVR4nO1di38aR5IGYZrdjBhymb158JgJh4ATZsz5BMhoUSQECMRLLwRycNYYyVawYkWSLct5Od5LsnvZ3di+7N7m373qngGGlywZWZD78Tm2BALRX1fVV1U9PR2dbowxxhhjjDHGGGOMMcYYY4wxxhhjjDHGGGOMMcYYY4wxxhhjjDHGGOP/A6web2x7ezsW83rDHvf0sIdzebC63VadzhuscHdWVpaW7t69u7Jy584nXCQwN+/1DHt0A8ITm8/JEdMnd/6Qv5OgeJ7X85Se11P4W0li2aU7JjnotQ57mG8Jz/asg7tzNyFJPObD6xugKEr9qocfsHc/EWa9wx7sxRGel7mVGnCj9G8CxbNLjMPpHvaQLwLPvIO5y2qsdjZDPZiSXRFmw8Me93nhzUVWWOnNtmsHGDKS+1Vw9AYytX7Ww0qjp3pzZyleWpqcG3lpdc9xtS56lB5kMxuKx5d3luPxeCjL6kmAdseotFQNjrawbgt3O+0nSVQoniqgaLVajUTgHzEaRa7kznIIRLbTjiyvv7c4wq5qnfuE5SlWsRvRDyob30kilEynplJTnMk0NcWY4A/DcJFq9FF0J57QS1rHhUzJ1wTnsIn0g8exwuubjsfz2XgaIVc1Y9uOecPhsGd7aioWmzI1wDAmoFnYAdGltHbnWWZ22FR6IybUGgaEOEssF6LVtH8blGPaE/bGoBYtc1OeMmfSgBgzilLxhFZ6eWk/N2wyveDkWNUSwC+UjkZMKZtbFy6XuFRqigB7aRvBBstIAaVCrEaZ+Hu50dOb4D5LKboJ/JJRDqLN47Yp0fcG4NAESy4n+EY1p+fvjJwVZ/cllaCUSIvADwaeSXFtPLC5+tPkRLTcclWw4rAptWP2jqTOfmI5yrXxAFYcBz6awgBXxehJljGJKN7IH5TkHym5Ce6rISjFUbWd3FTKCPDBHwU+n/IdJmvqYMlwKJ3g1dwhmUYoaTj31amndpDGgNyUkRBLp3eTyYJLRSGZ3E0rXIFmu71NTLVpRp4VRqalinGKyOjZZLRpFi5F6CVdyJX2LUOxFlKASzf8LGTK5C6xJRalFsUISrNEr/RUTRyRhsojKGmCSriqjBpRU9hCuy6UhtqMJw0wrlx4XikFeIliE6H4Mi4IdglJrR2rBVYxI39/NNTGuqCW2gmkEMT8wDVdrp0QizmpVZxiF9XYxOikYgWWSWxKriE9DFM1Nhw1MxKhGLyvWCaLIsoYp7B3ImNIf44GmJJ4NmTEJH1GNSSnwja0TCjyo+Gn4ZuSasEIg9WRwyLiSmd5Sn/eHphn40kcrSk1gq15lCWWBj8dgZQBPspiFyxUmalwnlMcNH7BDp+HMtaFdhUzcnkdV1B7MFYYekfsvIfbCVbaESGte/Kgn7sIWqILApd6VByh3SlMMRX2qDmD4lfmhkxwWiSawccRZ8vjBGFMFhLnXIDqhCSFXIQix+jmUSMBDduIwRXso2wCcb6wc4oQZHn2DVT6WpJngSI2YkyH7ijzxN+fHypBdxU7GCX5qilvKQIimixcdIlNAxYL8m6KGNGJ1CTDzgyVIZgQz3MIpbZ1GQ5rTKLH8tJ5wUP6jCMsN6nwNLpPciclMcOs3aYFMtFUIW3TlTBBlOD5B402/6LOSu25rSyfcqUYLKd55Xfr+aVhag0WUiwzUZOujHsIlOXZaV19jwwtzp47ISoEP+Xr8N4E2oUazmj1oiUSiSz70/CuwlkXcUFKsSgd9qRBZlBIvwc1yIM9YkJf51LhGwjyOs9DnXVPWnaBnk55ra6MUu3yQ3TT2D6WTTChTYdLGRTX858B772HoPR8KH4hgno8OeyD4h6VQEbQGpsuM6M4AbXy5rrG6vFuz8/Pb3svucjL1cjQXOnp7amUz7XMsxQwfPhQ94DXS+kLp/2HD/i9PZibFMkYujy6r3gBu3j2MNzbOfQoUDo4OCg50OL8JeZPT4TH8hdHZWtqypgkDQH/Kb9n1cHUZ40XzvuNRVOQU9AaJn9qUhhKkbMWwT2zqLK+ZldxbSOPcpfG0Rl/8HCPkgppa3nKl3ap0rlX1H0GJlyOXywMNcgiI47E7W1EfITn753RQwVRCehda8G+ZkOXdeljoaZ78DnkwrLVmMIySind36egNBR7nX371I+MRtxFTUXvg4vsffbZ3b75wiNU1rT0VI6n0UsRJ3f1c90D3Z7x1rRz1ZfcwS4WZxvOlk1f+PphE9IONBm4jYoEJP6zT+u6voHoRev2LoKY4yG6jFovtjxdd9f3UF4H3asLunZpOdscZXz5rZ0U3uzypUgbNVPb89Cf6j7t0wd70WEvfsSMk5dQJwTvfqovPoyjsHfV6IrrQXJaoSfthN6eIM4XxIhMdGXv8+m9+kOup9N5+hIEitdOAwMHI7S+1B6fjuj8PiPCRXOyZTY+eeFcoQGf3CVGZKoMr2fhQ+5s9/h86+J6X4IYA1MkNSlM97Z71ZhchtgrhFo1N184V1HKsr0qO4pfdhGtMUWiLLkMuRLsMYDZ0zMJXrOfDrhS5+F40vm6y6tgQl5a1ibAROF8xgple9bnkC+M5DojqpEJWOox1jBaO5MgwDGY3HjxQr5kzOhSxjSQY1FWY41EsvngLEnll9vDtflatGskbhpVSr/aQvfnV872UYw1NND1cie+3suibc+qzxXi9WBCDZdWTPbZeKFA2mmvC+LN511Ea5hqmvycXewKqTB6Ez/w03XHIAyD0NxQIdVJgWpIW6UlVIbS2YtSEmQZzQxQjXYEfq8R1zWmiNLpS0JXugjmsQk7zbjW8dRAVz5yECL8clXHGXch28cLksYf2XhS4Ssth8604e7jJ5qJYVEji7LgpkRrEHm/FOkqNX/asOPMjvkcHTRZgfistxlxELFZwAx350FJXSE9b4xrCIaSZZdyuRRy5FkFuOQqRrKaxwX/sloK7SCjD18bji5j5+dNnbbwRBUbXWtjuFax26taAVpDA7RTkzU81eFt7KRYZ1rDDlXNBpRQbXhmH0yhYhFplubQlv+JcvUG1NSH3bSaJgz3Yx2fvp0hrETCcL/JULbbuQ2tEeUB3PQniKAQspZ8oKR8qNC4hq9nE6huoZNZsisjnqmelRiBIZ3fbFofHlpKqzhFsvpCkqhpBJF9fXc6u4sgsdsaseQG1wjJNeGa3X9kv2ZvVKv2xwMkDOjA+XhVB+k+Ds7ow9eY+EQC+qZbx7SB9hOvlXaPbU/7r72BPtUt5kjTkXl0bKHLCFcO/DJSAzFBUn7nQOeUXEEE9UUBOG0cHBHG9sfr9rUK9/jIrgTiANc9omCcnYp7F3IFy/tglFJiE+1IfPyUNhjow02J+Jq5jvrnC8zQYKm7nqqeLF0/sRjoIjLi3TUo7eOI1PC9iprcYYvhGgImhXXuwI6/OziwH2bWHyMyBfbDAaRmhqX45GmYhKGUDEn6J6h8vCqxyGwwwLCJzD89pen80/5aQ+EXW8zVJ+rC4WqZxo9tyJeVlpFvysQoOb/F0OpWMmPuiHghIs09WnsB1NbAlPB4/bH94Mhu3yDkB2K4CHGIMnmfsUBJlCubQCUzXbwuPbHBIMGIDORy6RbYxIz6p8SEMh3mzPUsvtiovtlM122okEK4cGOqKQkzbDibJzPhCOQCiz9tqAw31v2ZwouDzNHRYRUYvrCv++2mF8DNtTGoly7WKAoxKaMxLYWWUQId0phN9joeM4z6mChsHbieuNh2K7Jahhb8YvoQbSYkPf/UTyvvpivJZKHA4e1SuHag7jYG6gZX3NjYWFMTeyFZfXy0xhwx3xz4H29gWvajir2AuRUIw/wASpO7SyWihGEKJVEyg8dGc9dPGmNknkohlxlbpLTZJjaSsdkdA0MaYKFpwzraDLGhqvJus0Xc9RnTKMIwHJLg3c117+n2nhBtYC/1H0XWiHzaq0eYYRVXA4gwrHSmmQsgeJengKEvjcsPhIi/0evki4GmzfUTlI1ztMUMT9x8IrUxRCGJ2JFPoGLZny4UkqbSSfHEhHaVtwMQ2Y+CqlEUTcVZzcp+DG20KKqu+HidUZ60c0f2Ddl+4L9mP1JKgkEy/vb+8k60Wk278D4DlLxVxv5WP8RCWrftIpRMosL10rGBJlKi8VMp7QeKkPMoKetCTPm4WDw+PIig9En9pKQSNCOyqyiNGK6UjzqWNNcutpGydkgYFbDg2A8egxHJ48y6fUOwX2MKJkTSxZHw9gR1uQDe8lupkrHAjOMkAc4JXllGpWOzgbZEdiPrCBKABXLeJtvsjaTVkyKCBEFJcYTq4KMAcFXzCbpepNsZ+gqnDGMrZQIVjWCEZSSX1g8xJfBKIid++4GrsA/BWTqwb0AU2jeOXpBkcTrAXo5gwOQvcRlbJKls4vK5DMTDzAYuUqctSsaI0JYTtFsEX/WjbGNzrbRZpuuR6yF9KB1qeiV2bcMBWrcoBFWGxvREicvbuIyjbRnD7XXOLgRw3FUOMY8jEwlCbM1SI00QH90YwEk9YsZky3AlmzoUcNQ6GR2dL5kt6pBPGNpMmw8Q5ABQyyfq1W9+0wZ2O3Rdf5KgtAzhDcfGJsO08lujeZhFP5cRO5sLK1YSheG1o0IjNKEI32g1jvbKAEo6WwF2HGcrNRm6jpXR1enWiJXUWDyxgNPWS6CWeFeUkhNoSzHvcrUzxE7e5qVGY6HE4A0CXCbQOQKnA/unkviPWqSg+m5+fxh9+7Uot5gBdhxjy7gaDG8pqb59yMq4LfhZC87jaDOehZxgUekcI4Ol+/VahslTP55KE+Pokn0B6rKzFjLAmgMsYgQrJgacx2+rJhsMd09p7RBJsdLBFOxW9hfUzKKEnmELo3taGgx35RJ8SoYzZSqdYwhr80YPrIkDyIxVyMDnmmB2C+kGQ2NVy9Bg3tp6Btja6jAnqE6x2KBufnbti2++/OaLr54/6yDZYJgW8+App2DE7nVvp3jmaptjkO1UXtlEwjBvQy2GSGuyZ/Yvvvza9/WX37ywPzNbNMMH41qaBO3foMq8c76CfN8+f6Z5u7nJ0BjF0Z6HLsPR3cvOyX3p2dcWB9ovNpsxcXmI/3y+ORJg2OJhfnbkym2HPeFYsHr92y4DNWhsTc0qhrFui76v7L1saEQ28lHQR/XQxVylH8HDR4OtlU6oDG0ahj41XRALftFS93Bg9cVWNz/MsKjJck70rWaGNAzzGZwRTaZMr0stgUAvR7Vfy6MB6lGAWwCGIKU4WfhaDJvhtbUha1/uRMXeRqS1rwp7NT+yRFsMsZiCCbvzBcac0CU3dvuhmBvwen7YwZhMIKUmLUMjOm7G1357HvL2tKHBPA3+qWuNRTMNFmG3kWbzWLLh8zJy5zAI5lH72rf92qEcHcyAgJhsIskC/mq81HXS9FJz86VuD74tiO5lQ7PZqou5niCkOrT2RXSlxbCkMuyzfh0W5Q01L+LL+AePhF4Xqi4IJzDMQJLqy9DQeOXxH59sQkj0ZmiY1kUpPR9X63+tFNP5W50MmYk+BYrViSYODqEtPlovCehybpkOMi2GLS91lZsJ0awOxnqL4qUQmp7uE4eeVbx2oZYe2tfQtuuN31pqMBT6lmBWbzAw82hGnnOGL2mLwix8oL/ThsbrNk1FqrwwvIqbpM1tax+GMbzKxm+SsNHOAnRgri4b9meo0LwcbgrmMpghY+rPUA3EGO58pfgs3YMfngcnWUhTGLb/5AR12jDTz0vfBXKMwtCkaS2apbdqhBZDKiu3QsxSr5vpxkPahheD+U0cOu2hajlGGhsq2WKgK2UXZQjpsMmwGYi38hpTKZHo/SPeNZV1tZ7moO7ONBKn5WRTolg9FlPwUXMfhn414/fMh+8I4KUwr4wJ1zSturSNoSKnnltkb02yUZjSed8aXfz2qV1dciwi+HkiCkHU4b/wE21Ng6u2ylXelwBaqjDM5119GWKKVrzyzYaYhgvSp988s+qe0c+VGtRsicQlaRMqSHOHFFnqDZFGNqZv5f3u4AyYTApDWyHZ20sNip7Oxnleetr8gaWc+i/A1vPnWw1TPX0iWrulFhgqv3S3asOSDf2heIUEdV6o2nAcgiGrhb42NBusOg++qOE6bimN+CWmeM2u1nF08dQfbopMi6ilKKplRCWPO2Do8S+hUjk/3KLK0G/jminf18kQyrJpXXA1+zSpldJHX3/1/Lnd3iCDl7vNuGk046+4c1SWlIuCT3XSElnFqFzxXWxyhkwsLmw06aLSlfbMkBBckbqmHrOYM+jFc3tXx0hvffc+4Hv1Uf0UuSIpo68gK2FYka94r/d8hctghgykC02T31iYAWMYzFtAD1O00O3LNfRx8VlXr2H+7sYNzPDGD+TiQBElIxGx6kujPA7Dkly56lvYvA5cz+Dd2PlCssWw1T59//4NGPJ3W71W0izdT1q+J/wwxe/JNdLTMmfiXEkUsfkj8uSEv+a44rMyppXFRCym1eZyYmu1zfyDMuAbN77rUa+ZDV1rjpYf3m/gewvoKDMNGdeXRBiFqTs1nl+aueJ7LecqJA3jBTftSk2dpPaWRcAkPWtuQ8dKo+U79R03bmwBwZIVfjfoaDyUTbDqnbU1OXClZvQKDFY4JRCbaqrW3lqG33WtmSqon7TZ0fynGwR/2qLNKKKzTTEpYwqFoDNRD9rQ8/oVMeD0Xp3gVCpETMGS1YJmIUMZtVljQ0ubEel8dPfr/TupKJKDbQ5Mm7eegQDRFjM6VQga06h9owovLe1Hr+5GNrcQsClumtG2+UpKpLfev6HGYbsJzZbjeS8+cy9sjQU7Q1Td4ZDHBKcwweX2XQ0UuGr0CjXVOosm8e3XtnxV018o621mi+HZdz/88MP3zzqXLyzFRv28fdCQJY27gorO60rYgiAzOx1X/ym+NnO19wfFQOiq1copvoqvJMUUyLvilWacBXukBcMWemGwWqfrX/1ZYYjjsvEyvFvIa81wJnBRF767oX0jDl8Tr7Rygy4xlIjvpNLpZSmURgqioU1b2/WZLoY0998//vj7H3/88S+H+IoUFC/R06LKtSiLbg/Hkbv80u37N0BvpJUPr5Sg1buAb8PHlwMlvB0qiw+ESEg8i076LFmoDA//8nuC659bSG9RZVyoTOMrbRUIQW+Kwzeioqf6tt3RFHho5qeryxbbzmBO2K+RLST4pnuy25z81VNSFhV7rh02rFpHhOBfieyaUdKjs0VQ+SSP0Gkpk08BP3CIkL65ERBfG+f52v6jqzzkLCcs1dh++/H4ECqeZUU68zfM8G8c3t9VRkzZ452KILGSh15+iohV6xwe+BDwEX1thZsJXm1hGoyccb82Hz+TouVzF2b4R7It5bTKGVdXyaE8UymcAqFQa52lpKdqtaWVfcejXOzKD42KzSxJ+n63bkkhsgOsn5vSn/y54aR0pUryjG8XNbCj12wuqv20kJu9tFXei8HjyPT1U3zLud/c14yQ1P/2V3RINomVlVSaRraT8slx8RSFeK2ESvcWhnjwx7y4IvFaO2qGxrNPUBlXYH0o5kR134yhinZ3wYBYTOljtNPp+tK9mStOgVp4AvgIT40hW9OP7wJeRbZ6P1+l6ebS1EklWsUvhGQYqXVuCaf00lKkexPG1cE7N8PcB1mVJHwicGLpfvPQS3JmmxH5T8x0r8JGUwoovYb5JBBZ0ncWMWSqHpquciW4C9OxYGBRFISqIIiLuQV/mxGkxNPrkA2KBroPTZWiofgYbd7tfVyPtCTMDf3wPeu0B8Nt1S0saUeJawE+G19FrtNyEepUskPP0l7JQfVatxVcT3ExxGMTUvh+NnyQMrxXYmv3JheH6KNdCGZ63VohsaGnqwi5GFv5uFg3Nxa3sXca6if5RcedLE4Q+LDkGnF4fE6Wnq1BHhQnR+qEaM+s3LMIwAfNSvpE6N4+8yFCH6J5ZcfXsW22MoNyaLLGq2dkQfKbERz4MEzGVI0uzjkv+/SAARGGsWJlJTd0U6TY6uC5JAZfBufqanNVtp3Y5t5779XrxZq6KZNfynncXnwofcwbHsXz6D3zgsDcA0+rsSz8s7TUduMhzzILr4CQTKu7UMFJbS/fA8zdb/g2X7uMPQbvFJ5YMLewOANYXFhcarJjQTRWxCCm80poXeqn87cJQ4d/STkSCkJxXx76uVfngNVqhWrgD00vhdbnfmaRsHnv1YfNOsdMBxfwU7lKYCFT0+NKnl3JDPtQqPNiXlxiWYkcJMjDsBcXFhzvKQg292uY6bLCUBYcAUemVmImxLlRks4z4ZzNyRMTgUyNXfI7FhfkyQnhtkrxUfMKdx29UhhOTEzIAYcj+CsxXwtWb1CWFwILDmAw4ZhTGd5W9veZoXN6TZ5YwAxhOoCj89fFMZwDywSU4QND8ZVK8aVI9mLQtpzyeLH5koAsDnKHy1VjXgSjCBNNOIIqw/dey8dF2hIMKI9eiXLzNZOBgBD4lYSiNSdr6eFIi6pGvO2QxRnb4mJOiczbwqTmVYIcEOZGMdt3whqQHRMdUI34OhqQZdkhOwICicOXHTMBziqM8LH6DVQCitdpRx4gkfhalBuMZBFTDHZNxUQlOuqljW5WsaAgtvtpEMuMrHkCU3TIE50QAmKvgz5GCPPqqIVomxFxJEYD7c/cvi222ZC8QYAiYNhHep4Jr6jS6GA44Xj92tH+jLww206QWF3c1s0Jo3FEck+4xYAqj0K03ftkQehwyUm5/ZlJRIIUxXQ5x+hasRJoSImA2vlMLCx0PAGVTLuSImVykMcqT45qLDpb0tHFsFtVOjJFg6Hg0IWFq75YeE54tFrSyfDNQCplENN5OTqSebHN7VSGk5MOjMnJLpN1mbQ5J1GvTg5c5bbn82LbodFG0A3HgiAIYgMTlXMzFERwB3n0QtEd1ZoQGMq52xivCG6/nuguYLSY1ESuOA9+Ko5cGT4X0NYxDmC48J4GrzqzxRkMQU890Yo4Yn4aFuWcu5UEHVHRMdnOMNCDlwaiVpscOdExan4aCETl+ZYRJ4Gh2MZQfIMNRaSpeQQH9JijpadeQfTOabyUMHzVxvDsOJwQtXWeMGeNTgR637A2JMgBFJvTKo0YdQgvsdK8fPnyNSB3tgnxG9raYZgteWKE8n5MELaFtowngg1wroCWYVKQHQ75TQyjHbU6IBAdnZZfcDit7R0hNomsJPuukfdh2PXcCIlNDJL0fDfDiwACt/sNgeiorDEGHPPWDhN0ld5vw3AiMCKtYlgUp53tJrw4Q9ST4YhkjDmIlw6CF2Y40ZPhhDwSRnRHBU9scIZCL4ajEYnbDlkndzZHAnpTu3QuhhOBUZDTWdkZ7pb6CzLsMyNCYIBjZi4LbkfUE+we3uUwhJw6/C5qW46GO6PwbRj2eX4Esn5QEKLdbCaTb6i0O3DzZoNSx++Sr/ImZxXT820IABYWZNlRvVm9SfAfGMq/bbip/vhmX3S8VhAmHYGJeWcXZt+twjr/5+eff87/TPDvP6tf3w1+/s/ekN/t1dTZv//LsPGPd3sz28IHH/xOxQfkv0vGBz2/bcM/3u3d64sf/O6DIePvPc5pvzxYF//3X4eNX97p7evTN387dHwcfZcM3cJvfzNk/PbjxXfJcFq8ZIP8psdjYEHmUZnMbht2/98ELhHWmX9i/Nu58dEb8PFHv3z80Ue//PLPX35pPNMfH338C8bMO91h61zoDcdiP8ychQ/fBmgU2sY3Ajua9QxMt8HdjmEPfowxxhhjjDHGGGOMMcYYY4wxxhhjjDHGGGOMMcYYY4wxdP8HbSy1FAlq4jQAAAAASUVORK5CYII=",
        "suggestions": ["What's problem with you?", "account", "shipping", "product details", "order", "support"]
    },
    "customer_service.introduction": {
        "text": "🛒 We have a great selection of products like computer accessories, electronics, and home appliances. You can find items such as USB cables, HDMI cables, and steam irons in our product catalog."
    },
    "customer_service.product_details": {
        "text": "I'll provide you with detailed information about the product's technical specifications, pictures, and prices. What product would you like to get more information about?"
    },
    "customer_service.shipping": {
        "text": "🚚 We offer various shipping options, including fast and free delivery. You'll be notified about the estimated delivery time and can track the status of your order.",
        "suggestions": ["Do you need assistance with your shipping?", "change shipping address", "set up shipping address"]
    },
    "customer_service.returns": {
        "text": "We have a flexible returns policy. You can return the product within a certain timeframe and receive a full refund or exchange it for a different item."
    },
    "customer_service.support": {
        "text": "🔄 We have a flexible return policy. You can return the product within a certain timeframe and receive a full refund or exchange it for a different item."
    },
    "customer_service.say_goodbye": {
        "text": "📞 Our customer support team is always ready to assist you in resolving any issues related to your order or product. You can contact us through multiple channels."
    },
    "cancel_order": {
        "text": "🙋‍♀️ I'd be happy to help you with cancelling your order. To cancel your last order, you can visit your account page on our website and look for the 'Cancel Order' or 'Order History' section. There, you should be able to find the order you want to cancel and select the option to cancel it. If you have any trouble finding this or need further assistance, please feel free to contact our customer support team, and they'll be glad to walk you through the cancellation process. We're here to help ensure you have a smooth shopping experience.",
        "suggestions": ["What do you want me to help with? 🤗", "change order", "buy item", "track order" ]
    },
    "change_order": {
        "text": "🔄 I apologize, but it seems like you are having trouble changing or modifying your order. Our order management system does allow you to make changes to your order, such as adding, removing, or substituting items, before it is shipped. To make changes to your existing order, please log into your account on our website and look for the 'Manage Order' or 'Edit Order' functionality. There, you should be able to view the details of your order and make any necessary modifications. If you are still having trouble, feel free to reach out to our customer service team and they will be happy to assist you with updating your order. We want to ensure you get exactly what you need, so please don't hesitate to let us know how we can help.",
        "suggestions": ["What do you want me to help with? 🤗", "cancel order", "buy item", "track order" ]
    },
    "change_shipping_address": {
        "text": "🚚 I understand you're having trouble updating your shipping address for your order. Our order management system does allow you to change your shipping address before the order is shipped. To do this, simply log into your account on our website and look for the 'Edit Order' or 'Change Shipping Address' option. There, you should be able to update the address details. If you're having any trouble finding this or need further assistance, please don't hesitate to reach out to our customer service team. They'll be happy to walk you through the steps to successfully change your shipping address. We want to make sure your order gets delivered to the right place, so please let us know how we can help.",
        "suggestions": ["Do you need assistance with your shipping?", "set up shipping address"]
    },
    "check_cancellation_fee": {
        "text": "🔍 To check the cancellation or termination fees associated with your order, I would suggest reviewing the terms and conditions or contacting our customer service team. The specific fees can vary depending on the product or service, and the stage of the order. Our representatives will be able to provide you with the exact cancellation charges that would apply to your order if you decide to cancel. Please feel free to reach out to them, and they'll be happy to look up the relevant details and walk you through the process. We want to ensure you have all the information you need to make an informed decision."
    },
    "check_invoice": {
        "text": "📋 To access your past invoices, you can typically log into your account on our website and look for the 'Invoices' or 'Order History' section. There, you should be able to view and download PDF copies of your previous orders and their associated invoices. If you're having trouble locating a specific invoice, our customer service team would be happy to assist you. Just provide them with details like the approximate purchase date or order number, and they can look up the invoice for you. Please let me know if you need any other help accessing your invoice information."
    },
    "check_payment_methods": {
        "text": "💳 You can find our accepted payment methods by checking the 'Payment' or 'Checkout' section of our website. There, you should see a list of all the payment options we currently support, such as credit/debit cards, digital wallets, bank transfers, and more. If you're having trouble locating this information, our customer service team will be happy to provide you with the details on our accepted payment methods. Please let us know if you need any assistance checking what payment options are available."
    },
    "check_refund_policy": {
        "text": "💰 To review our refund policy and understand the circumstances under which you may be eligible for a refund, I recommend checking the 'Returns' or 'Refunds' section of our website. There, you should find detailed information on our money-back guarantee, the timeframe for processing refunds, and the specific criteria that would qualify an order for a refund. If you're unable to locate this information online, our customer service team would be happy to provide you with the details of our refund policy. Please let us know if you have any other questions about getting a refund."
    },
    "complaint": {
        "text": "🤔 I apologize that you've had an unsatisfactory experience with our business. Our goal is always to provide excellent service and products to our customers. If you would like to file an official complaint, I recommend contacting our customer service department either by phone, email, or through the contact form on our website. They will be able to document your concerns and work with you to try to resolve the issue. Please let me know if there is any other way I can assist you."
    },
    "contact_customer_service": {
        "text": "📞 To find the available hours for our customer service team, I recommend checking the 'Contact Us' or 'Customer Support' section of our website. There, you should be able to see our customer service hours, as well as the various ways you can reach out to us, such as by phone, email, or live chat. If you're unable to locate this information online, you're welcome to call our main customer service number and one of our representatives will be happy to provide you with the current operating hours. Please let me know if you need any other assistance contacting our customer support."
    },
    "contact_human_agent": {
        "text": "🤖 To speak with one of our customer service agents, the quickest way is to call our main customer service phone number. Our agents are available during our standard business hours, which you can find listed on our website. Alternatively, you can try reaching out via our live chat feature, also available on our website during business hours. If you're having trouble locating the right contact information, please let me know and I'll do my best to guide you to the appropriate channel for speaking with a live agent. I'm happy to assist further, but cannot directly transfer you to a human agent from this interface."
    },
    "create_account": {
        "text": "🙂 To create a new account with our service, simply visit our website and look for the 'Sign Up' or 'Create Account' button or link. This will guide you through the registration process, which typically involves providing some basic personal and contact information. If you have any trouble or need assistance during the sign-up, feel free to reach out to our customer support team and they'll be happy to walk you through it step-by-step. Let me know if you have any other questions!",
        "suggestions": ["Is there anything else I can help you with your account?", "delete account", "switch account"]
    },
    "delete_account": {
        "text": "📝 To delete your account with our service, you can typically find the option to do so in your account settings or profile page. If you're unable to locate it there, you're welcome to reach out to our customer support team who can guide you through the account deletion process. They'll be able to confirm the necessary steps and ensure your account is properly closed. Please let me know if you need any assistance navigating the account deletion procedure.",
        "suggestions": ["Is there anything else I can help you with your account?", "create account", "switch account"]
    },
    "delivery_options": {
        "text": "🚚 To view the available delivery options for your order, simply go to the checkout page on our website. There you'll see a section that lists all the different shipping methods and timeframes we offer, along with the associated costs. Feel free to review the options and select the one that best suits your needs. If you have any trouble finding or understanding the delivery choices, our customer service team would be happy to provide more information and guidance. Just let me know if you need any assistance navigating the delivery options.",
         "suggestions": ["What do you want me to help with? 🤗", "cancel order", "change order", "buy item" ]
    },
    "delivery_period": {
        "text": "📦 To check the expected delivery timeline for your order, you can usually find that information on the order confirmation page or in your account's order history. There you'll see the estimated delivery date range based on the shipping method you selected. If you can't locate that detail, you're welcome to reach out to our customer support team and they'll be happy to look up the projected delivery period for your specific order. Just let us know the order number or other relevant details, and we'll provide that delivery timeline information."
    },
    "get_invoice": {
        "text": "📋 To retrieve your past order invoices, you can typically access them through your online account. Look for a section labeled 'Orders' or 'Purchase History' where you should be able to view and download PDF copies of your invoices, even for purchases made months or years ago. If you're unable to locate your invoices there, you're welcome to reach out to our customer service team. Provide them with details like the approximate purchase date or order number, and they'll be happy to research your account and send over the requested invoice(s). Let me know if you need any other assistance accessing your past order documentation."
    },
    "get_refund": {
        "text": "💰 To initiate a refund, I'd recommend first reviewing the return/refund policy associated with your purchase. This information is typically provided during the checkout process or available on the company's website. If you're eligible for a refund, you can usually start the process by contacting the company's customer service team, either by phone, email, or through an online form. Be prepared to provide details like your order number, date of purchase, and reason for the refund request. The customer service representatives will then guide you through the necessary steps to successfully process the refund back to your original payment method. Let me know if you have any other questions about getting a refund - I'm happy to provide additional assistance."
    },
    "newsletter_subscription": {
        "text": "📰 To subscribe to our company newsletter, you can typically do so by visiting our website and looking for a 'Newsletter' or 'Sign Up' section, where you can enter your email address. Once subscribed, you'll begin receiving our latest updates and content.\n\nTo unsubscribe, there is usually an 'Unsubscribe' link included at the bottom of each newsletter email that will allow you to manage your subscription preferences and opt out. Alternatively, you can reach out to our customer service team for assistance with updating or canceling your newsletter subscription.\n\nPlease let me know if you need any further help accessing or modifying your newsletter subscription status. I'm happy to provide additional guidance."
    },
    "payment_issue": {
        "text": "💳 If you're experiencing any issues with your payments, I'm happy to help. The best way to report and resolve a payment problem is typically to reach out directly to our customer service team. You can contact them by phone, email, or through an online support form on our website. Provide them with details about the specific issue you're facing, such as the transaction date, order number, payment method used, and any error messages or problems you encountered. Our team will then review the details and work with you to identify the root cause and get the issue corrected. Please don't hesitate to get in touch - we're committed to ensuring a smooth payment experience and will do our best to resolve any problems you may be having.."
    },
    "place_order": {
        "text": "🛒 To make a purchase on our website, simply browse our selection of items, add the ones you'd like to buy to your cart, and then proceed to checkout. During checkout, you'll be prompted to provide your shipping information and choose a payment method. We accept a variety of payment options, including credit/debit cards, PayPal, and digital wallets. Once you complete the checkout process, your order will be processed and shipped to the address you provided. If you need any assistance navigating our website or completing your purchase, don't hesitate to reach out to our customer service team - they'll be happy to guide you through the ordering process step-by-step. Let me know if you have any other questions!",
        "suggestions": ["What do you want me to help with? 🤗", "cancel order", "change order", "track order" ]
    },
    "recover_password": {
        "text": "🔑 If you've forgotten your password, the best way to recover access to your account is to use our password reset functionality. On our website, look for a 'Forgot Password' or 'Reset Password' link, typically located near the login area. Click on that, and you'll be prompted to enter the email address associated with your account. We'll then send you instructions on how to reset your password securely. This usually involves creating a new password that you can then use to log back into your account. If you have any trouble with the password reset process, don't hesitate to reach out to our customer support team for assistance. They'll be glad to help you regain access to your account.",
        "suggestions": ["Is there anything else I can help you with your account?", "delete account", "create account", "switch_account"]
    },
    "registration_problems": {
        "text": "📝 I apologize for any difficulties you've encountered with the registration process. If you're experiencing issues signing up or notice any errors during registration, the best thing to do is to reach out to our customer support team. You can contact them via email, phone, or by submitting a support ticket on our website. Provide as much detail as possible about the specific problem you're facing, such as any error messages you're seeing, the steps you took leading up to the issue, and your contact information. Our support staff will review the details and work quickly to investigate and resolve the registration problem. We're committed to ensuring a smooth sign-up experience, so please don't hesitate to report any issues you run into. We're here to help get you successfully registered."
    },
    "review": {
        "text": "📋 We greatly appreciate you taking the time to provide feedback about our services. The best way to leave a review is to visit our website and look for a 'Leave a Review' or 'Submit Feedback' section, often found in the footer or customer support area. There you can fill out a short form to share your thoughts, comments, or rating of your experience with us. We carefully read and consider all feedback received, as it helps us continue improving our products and services for customers like yourself. Thank you in advance for your review - we value your input and look forward to hearing from you!"
    },
    "set_up_shipping_address": {
        "text": "📦 To set up a new or different shipping address, you can typically do so through your account settings on our website. Look for a section labeled 'Shipping Addresses' or something similar, where you'll be able to add, edit, or select a new address. Simply fill out the required fields with the new address details, and save the changes. If you're having trouble getting the new address to be accepted, double-check that you've entered all the information correctly, including the full street address, city, state/province, postal code, and country. If you continue to run into issues, don't hesitate to reach out to our customer support team - they'll be happy to assist you in successfully updating your shipping address in our system.",
        "suggestions": ["Do you need assistance with your shipping?", "change shipping address"]
    },
    "switch_account": {
        "text": "🔒 To switch to a different user account, typically you would look for a 'Switch Account' or 'Change User' option, often located in the top right corner of the interface or under account settings. This will allow you to log out of the current account and log into a new one. Make sure you have the credentials (username and password) for the account you want to switch to. If you're having trouble finding where to switch accounts or are unable to access your desired account, I'd recommend reaching out to our customer support team for assistance. They can help guide you through the account switching process to get you logged into the correct profile.",
        "suggestions": ["Is there anything else I can help you with your account?", "delete account", "create account"]
    },
    "track_order": {
        "text": "📦 To track the status of your order, you can typically do so by logging into your account on our website and looking for an 'Order History' or 'Track Order' section. There, you should be able to enter your order number or other identifying details to view the current status and estimated delivery date of your shipment. If you're unable to locate this information in your account, you can also contact our customer service team and provide them with your order details - they'll be happy to look up the status of your order and give you an update on the expected arrival time. Just let me know if you need any assistance tracking a specific order.",
        "suggestions": ["What do you want me to help with? 🤗", "cancel order", "change order", "buy item" ]
    },
     "track_refund": {
        "text": "💳 To check the status of your refund, you can typically log into your account on our website and look for a section related to orders, payments, or refunds. There, you should be able to view the details of your refund request, including the date it was processed and the status. If you're unable to locate this information online, you can also contact our customer service team directly. Provide them with the details of your original order and refund request, and they'll be happy to look into the status of your refund and provide you with an update. Just let me know if you have any trouble accessing your refund information, and I'll do my best to assist you further."
    }, 
}



chatbot_data = MyChatbotData(training_data, 'patterns', answers)
UNK = "I don't know"

def exact_match(query):
    intents = chatbot_data.get_intents()
    for intent in intents:
        phrases = chatbot_data.get_phrases(intent)
        if query in phrases:
            return chatbot_data.get_answer(intent)
    return UNK

def preprocess(text):
    text = ascii_normalize(text) or text
    text = emoji_normalize(text) or text
    text = emoji_isolate(text) or text
    text = remove_punctuation(text) or text
    return text

def remove_punctuation(text):
    return punct_re_escape.sub('', text)

def rule_based_classifier(query):
    preprocessed_query = preprocess(query.lower())
    return exact_match(preprocessed_query)

def fuzzy_matching_1(query):
    intents = chatbot_data.get_intents()
    for intent in intents:
        phrases = chatbot_data.get_phrases(intent)
        match, score = process.extractOne(query,  phrases )
        if score > 90:
            return chatbot_data.get_answer(intent)
    return UNK

def fuzzy_matching(query):
    preprocessed_query = preprocess(query.lower())
    return fuzzy_matching_1(preprocessed_query)