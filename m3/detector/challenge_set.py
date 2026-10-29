"""Non-Faker challenge set for the PII detector.

Usage: python -m m3.detector.challenge_set
Writes data/m3/challenge_set.jsonl.

Every value below is hand-written and fictional. Phone numbers use reserved fictional ranges only
(North American 555-0100..0199, UK 020 7946 0xxx, Australian 02 5550 xxxx); e-mail domains are the
reserved example.com / example.org / example.net (and their subdomains); street names are invented.
Values are placed into templates that differ from the four M2 templates ("My name is X." etc.) and
combined with a BANKING77 query from the M3 *train* split as the carrier (never the test split).
Items are split 50/50 into challenge_dev (used to choose a detector) and challenge_heldout
(reported once, for the chosen detector), stratified by category and format tag.
"""
import collections
import random
import re

from m3.common import DATA, DATA_M3, SEED, load_splits, norm, read_jsonl, write_jsonl

PERSON = {
    "south_asian": ["Priya Raghunathan", "Arjun Mehrotra", "Sanjana Iyer", "Vikram Chaudhary",
                    "Nandini Subramaniam", "Rohan Deshpande", "Kavya Venkatesh", "Imran Siddiqui"],
    "east_asian": ["Wei Zhang", "Mei Lin Huang", "Hiroshi Tanabe", "Yuki Morimoto",
                   "Ji-woo Park", "Seo-yeon Kim", "Haruto Kobayashi", "Xiu Ying Chen"],
    "west_african": ["Chinedu Okafor", "Adaeze Nwosu", "Kwame Mensah", "Abena Owusu",
                     "Oluwaseun Adeyemi", "Folake Balogun", "Kofi Asante", "Ngozi Eze"],
    "hispanic_accent": ["José Ramírez-Ortega", "María Fernández", "Ana Sofía Gutiérrez", "Joaquín Peña",
                        "Lucía Martínez-Soto", "Andrés Muñoz", "Ximena Domínguez", "Íñigo Castañeda"],
    "irish": ["Siobhán O'Sullivan", "Seán Ó Briain", "Aoife Ní Dhomhnaill", "Pádraig Ó Conaill",
              "Niamh O'Donnell", "Ciarán Mac Cárthaigh", "Róisín Ní Fhlannagáin", "Eoghan O'Mahony"],
    "arabic": ["Yousef Al-Hashimi", "Layla Haddad", "Omar Abdel-Rahman", "Fatima Zahra Benali",
               "Karim El-Masri", "Noor Al-Farsi", "Tariq Mansour", "Huda Qassem"],
    "slavic": ["Katarzyna Wiśniewska", "Dmitri Volkov", "Miroslav Novák", "Zofia Kowalczyk",
               "Aleksandr Petrenko", "Jelena Marković", "Bogdan Horvat", "Milena Dvořáková"],
    "vietnamese": ["Nguyễn Thị Lan", "Trần Văn Minh", "Phạm Quốc Bảo", "Lê Thu Hà",
                   "Hoàng Đức Anh", "Võ Thanh Tâm", "Đặng Mỹ Linh", "Bùi Gia Huy"],
    "lowercase_name": ["tom becker", "linda okoro", "sam whitfield", "grace delgado",
                       "omar hassan", "jenny cho", "mark dunleavy", "rosa ibarra"],
    "first_name_only": ["Marguerite", "Tobias", "Ingrid", "Desmond", "Anika", "Callum", "Beatriz", "Lorenzo"],
    # the title is kept in the text but is not part of the labeled span
    "title_name": ["Dr. Okonkwo", "Ms. Halvorsen", "Mr. Delacroix", "Mrs. Abernathy",
                   "Dr. Lindqvist", "Prof. Ramaswamy", "Ms. Featherstone", "Mr. Kowalski"],
    "name_is_word": ["April Hood", "Hope Castellano", "Chase Whitmore", "Sierra Lindgren",
                     "Paris Okonjo", "Jordan Ellsworth", "Summer Vance", "Brooke Hollis"],
}
PERSON_TEMPLATES = [  # (template, placement relative to the carrier query)
    ("This is {v} from payroll.", "prefix"), ("Please ask {v} to call me back.", "suffix"),
    ("{v} here,", "prefix"), ("Regards, {v}", "suffix"), ("I am writing on behalf of {v}.", "prefix"),
    ("The account holder is {v}.", "suffix"), ("Hi, it's {v}.", "prefix"), ("Can {v} get an update?", "suffix"),
]

EMAIL = {
    "dotted_email": ["maria.l.fernandez@example.com", "k.mensah@example.org", "hiro.tanabe@example.net",
                     "a.b.nwosu@example.com", "linda.okoro.work@example.org", "j.park.88@example.net",
                     "tom.becker@example.com", "s.o.sullivan@example.org"],
    "plus_email": ["priya+bank@example.com", "wei.zhang+cards@example.org", "kofi+alerts@example.net",
                   "rosa.ibarra+statements@example.com", "dvolkov+travel@example.org", "noor+1@example.net",
                   "callum+refund@example.com", "anika.s+billing@example.org"],
    "subdomain_email": ["wei.zhang@mail.example.org", "omar@billing.example.com", "ingrid@eu.mail.example.net",
                        "desmond.k@staff.example.org", "huda.q@mx.example.com", "tobias@uk.example.net",
                        "layla@corp.mail.example.org", "milena.d@student.example.com"],
    "uppercase_email": ["Tom.Becker@Example.com", "ADAEZE.NWOSU@EXAMPLE.ORG", "Kwame.Mensah@Example.NET",
                        "GraceDelgado@EXAMPLE.com", "JPENA@example.org", "Sam.Whitfield@Example.Org",
                        "BEATRIZ@EXAMPLE.NET", "Yuki.Morimoto@EXAMPLE.COM"],
    "digits_email": ["sam1987@example.net", "jenny.cho2004@example.com", "k0fi.asante@example.org",
                     "mark77@example.net", "user4417@example.com", "lorenzo.88@example.org",
                     "r2d2fan.bogdan@example.net", "z.kowalczyk91@example.com"],
    "paren_email": ["folake.b@example.com", "tariq.m@example.org", "karim.elmasri@example.net",
                    "ana.sofia@example.com", "eoghan@example.org", "imran.s@example.net",
                    "kavya.v@example.com", "marguerite@example.org"],
    "punct_email": ["niamh.odonnell@example.com", "rohan.d@example.org", "seo.yeon@example.net",
                    "jelena.m@example.com", "vikram.c@example.org", "aleks.p@example.net",
                    "fatima.benali@example.com", "chinedu@example.org"],
    "org_net_email": ["support.ticket@example.org", "lanvu@example.net", "ximena.d@example.org",
                      "pq.bao@example.net", "ha.le@example.org", "duc.anh@example.net",
                      "thanh.tam@example.org", "gia.huy@example.net"],
}
EMAIL_TEMPLATES = [("You can email me at {v}.", "suffix"), ("Send the statement to {v}.", "suffix"),
                   ("Reply to {v} please.", "prefix"), ("Write to me via {v} if needed.", "suffix")]
EMAIL_SPECIAL = {
    "paren_email": [("Contact me (email: {v}) about this.", "prefix"), ("(Reach me at {v})", "suffix")],
    "punct_email": [("Email {v}!", "suffix"), ("Is {v}; the right contact?", "prefix"),
                    ("Use {v}, not my old one.", "prefix"), ("Can you write to {v}?", "suffix")],
}

PHONE = {
    "paren_phone": ["(317) 555-0142", "(212) 555-0101", "(503) 555-0187", "(404) 555-0123",
                    "(617) 555-0166", "(702) 555-0110", "(919) 555-0178", "(612) 555-0134"],
    "dot_phone": ["317.555.0142", "646.555.0119", "415.555.0102", "303.555.0171",
                  "206.555.0158", "480.555.0136", "713.555.0193", "860.555.0125"],
    "space_phone": ["317 555 0142", "512 555 0109", "305 555 0164", "602 555 0181",
                    "971 555 0147", "443 555 0112", "216 555 0190", "801 555 0153"],
    "intl_phone": ["+1 317 555 0142", "+1 (212) 555-0118", "+1-415-555-0177", "+1 646 555 0140",
                   "+1.303.555.0105", "+1 206-555-0186", "+1 (713) 555 0129", "+1 919 555 0161"],
    "ext_phone": ["1-800-555-0199 ext. 23", "1-888-555-0104 x12", "(800) 555-0122 ext 7", "1-877-555-0150 ext. 401",
                  "212-555-0137 ext. 5", "1-866-555-0183 x3", "(888) 555-0116 extension 9", "1-800-555-0172 ext. 88"],
    "nosep_phone": ["3175550142", "2125550168", "4155550103", "6175550194",
                    "5035550127", "7025550159", "9195550115", "6125550148"],
    "uk_phone": ["+44 20 7946 0123", "020 7946 0958", "+44 (0)20 7946 0412", "020 7946 0777",
                 "+44 20 7946 0034", "020-7946-0601", "+44 20 7946 0299", "(020) 7946 0845"],
    "au_phone": ["02 5550 1234", "+61 2 5550 4321", "(02) 5550 8812", "02 5550 0097",
                 "+61 2 5550 6650", "02-5550-3141", "(02) 5550 2718", "+61 (2) 5550 9090"],
}
PHONE_TEMPLATES = [("Call me on {v}.", "suffix"), ("My mobile number is {v}.", "prefix"),
                   ("Text {v} when it's done.", "suffix"), ("You can reach me at {v}.", "suffix"),
                   ("Best number: {v}.", "prefix")]

ADDRESS = {
    "unit_first": ["Apt 4B, 1200 Maplecrest Ln, Dunmore, OH 43210", "Unit 12, 85 Quillfeather Rd, Brambleton, IN 46201",
                   "Suite 300, 4410 Copperleaf Dr, Ashby, MN 55301", "Apt 7, 902 Thistledown Ln, Riverton, UT 84065",
                   "#5, 61 Wrenhollow St, Fernhill, OR 97201", "Apt 2C, 330 Bellmoor Ave, Kestrel Falls, NY 12545",
                   "Unit B, 17 Saffronfield Ct, Larkin, TX 75001", "Apt 19, 2208 Hazelbrook Blvd, Elmstead, GA 30301"],
    "state_spelled": ["88 Quillfeather Road, Brambleton, Indiana 46201", "4021 Ambergrove Lane, Dunmore, Ohio 43215",
                      "15 Moss Hollow Drive, Pinebrook, Colorado 80202", "702 Starling Way, Ashby, Minnesota 55302",
                      "391 Cobblefen Street, Larkin, Texas 75002", "56 Juniper Reach, Elmstead, Georgia 30303",
                      "1120 Brackenridge Avenue, Fernhill, Oregon 97202", "9 Lanternwick Court, Riverton, Utah 84066"],
    "state_lowercase": ["77 Willowmere Ave, Fernhill, or 97203", "212 Dovecote Ln, Larkin, tx 75003",
                        "640 Peregrine St, Pinebrook, co 80203", "18 Fallowmere Rd, Elmstead, ga 30304",
                        "3305 Rushwick Dr, Kestrel Falls, ny 12546", "47 Gorsefield Way, Ashby, mn 55303",
                        "1501 Tidewater Ct, Riverton, ut 84067", "26 Sparrowgate Blvd, Brambleton, wa 98101"],
    "no_zip": ["315 Larchmont Way, Pinebrook, CO", "48 Ivybridge Lane, Dunmore, OH",
               "2290 Kingfisher Road, Larkin, TX", "6 Ashgrove Terrace, Elmstead, GA",
               "731 Fernwhistle Ave, Riverton, UT", "104 Marigold Bend, Ashby, MN",
               "59 Stonechat Drive, Kestrel Falls, NY", "870 Heronbrook Pl, Fernhill, OR"],
    "zip4": ["1450 Copperleaf Dr, Ashby, MN 55301-1234", "22 Rookwood Ln, Dunmore, OH 43210-4567",
             "309 Bluebell Ct, Larkin, TX 75001-0099", "5120 Oakhaven Rd, Pinebrook, CO 80202-3310",
             "71 Linnet Way, Elmstead, GA 30301-2288", "640 Sandpiper St, Riverton, UT 84065-1402",
             "13 Gladewater Ave, Fernhill, OR 97201-5521", "908 Briarcliff Blvd, Kestrel Falls, NY 12545-7710"],
    "po_box": ["PO Box 4417, Millbrook, NY 12545", "P.O. Box 92, Larkin, TX 75001", "PO Box 1180, Ashby, MN 55301",
               "P.O. Box 305, Elmstead, GA 30301", "Post Office Box 77, Riverton, UT 84065",
               "PO Box 2026, Fernhill, OR 97201", "P.O. Box 58, Dunmore, OH 43210", "PO Box 9931, Pinebrook, CO 80202"],
    "rural_route": ["Rural Route 2, Box 15, Hollins, KS 66845", "RR 1 Box 220, Pell City, AL 35125",
                    "Rural Route 4, Box 8, Cedar Bluff, IA 52203", "RR 3, Box 117, Marshfield, MO 65706",
                    "Rural Route 1, Box 64, Dunmore, OH 43210", "RR 2 Box 9, Larkin, TX 75001",
                    "Rural Route 5, Box 301, Ashby, MN 55301", "RR 6, Box 42, Elmstead, GA 30301"],
    "uk_address": ["Flat 3, 12 Larkspur Close, Leeds LS6 2QT", "27 Hawthorn Mews, Bristol BS8 4RT",
                   "Flat 9, 4 Primrose Gardens, Manchester M14 5PL", "118 Kestrel Rise, Norwich NR4 7JX",
                   "6 Foxglove Crescent, York YO24 1QS", "Flat 1A, 33 Elderberry Row, Cardiff CF10 3NB",
                   "52 Wychwood Lane, Oxford OX2 6HT", "2 Cobble Yard, Edinburgh EH3 9DN"],
    "ca_address": ["45 Birchwood Cres, Halifax, NS B3H 1Z9", "1208 Tamarack Rd, Calgary, AB T2N 4L7",
                   "Unit 6, 300 Loonfield Ave, Winnipeg, MB R3T 2N2", "88 Spruceview Dr, Victoria, BC V8W 1P6",
                   "17 Cedarwick Lane, Moncton, NB E1C 4B6", "640 Maplehurst St, Regina, SK S4S 0A2",
                   "2105 Aspenridge Blvd, Ottawa, ON K1N 6N5", "9 Harbourlight Way, St. John's, NL A1C 5S7"],
    "no_commas": ["902 Thistledown Lane Apt 7 Riverton UT 84065", "41 Bramblewood Rd Dunmore OH 43210",
                  "3300 Gullhaven Dr Larkin TX 75001", "76 Rosethorn Ct Ashby MN 55301",
                  "1219 Pebblebrook Ave Elmstead GA 30301", "58 Windrush Way Fernhill OR 97201",
                  "605 Cloverleigh St Pinebrook CO 80202", "12 Marshwren Pl Kestrel Falls NY 12545"],
    "split_in_at": ["12 Sorrelwood Court in Eastvale, Oregon", "300 Fennel St at Larkin, TX",
                    "403 Quarrystone Rd in Dunmore, Ohio", "88 Heathergate Ln in Pinebrook, CO",
                    "unit 4 at 715 Millrace Ave, Ashby", "2201 Ravensworth Blvd in Elmstead, Georgia",
                    "39 Lindenfold Way in Riverton, UT", "flat 2 at 60 Seabrook Row, Norwich"],
}
ADDRESS_TEMPLATES = [("Please send the new card to {v}.", "suffix"), ("I live at {v}.", "prefix"),
                     ("My mailing address changed to {v}.", "suffix"), ("Ship it to {v} please.", "suffix")]
ADDRESS_SPECIAL = {"split_in_at": [("I moved to {v}.", "suffix"), ("Our flat is {v}.", "prefix")]}

# Name-like or PII-like words that are not PII under the labeling rubric
NEGATIVES = {
    "brand_name": ["My Chase card was declined.", "Can I link my Wells Fargo account?",
                   "Is my Mercedes-Benz car loan visible here?", "Does the Barclays transfer arrive today?"],
    "month_word": ["I was charged twice in April.", "My card expires in June.",
                   "The top up from May is still pending.", "Why was there a fee in August?"],
    "place_word": ["Can I use Apple Pay in Sydney?", "I am travelling to Paris next week.",
                   "Will my card work in Jordan?", "I withdrew cash in Lincoln yesterday."],
    "phone_word": ["My phone is broken and I cannot open the app.", "I lost my phone with the banking app on it.",
                   "Can I change the phone number on my account?", "The verification text never reached my phone."],
    "address_word": ["I moved to a new address.", "How do I update my address?",
                     "Do you need proof of address?", "The card went to my old address."],
    "email_word": ["I did not get the confirmation email.", "Can you email me the statement?",
                   "My email changed last month.", "Is the email from support real?"],
    "number_word": ["My card number starts with 4.", "I sent 555 dollars to my landlord.",
                    "The transaction ID is 20231101.", "Why do I have 3 pending payments?"],
    "word_name": ["Hope you can help with my refund.", "Will you chase the missing transfer?",
                  "Mark this payment as disputed.", "Grace period for the overdraft?"],
}

FORMAT_TYPES = [("PERSON", PERSON), ("EMAIL", EMAIL), ("PHONE", PHONE), ("ADDRESS", ADDRESS)]
TEMPLATES = {"PERSON": PERSON_TEMPLATES, "EMAIL": EMAIL_TEMPLATES, "PHONE": PHONE_TEMPLATES, "ADDRESS": ADDRESS_TEMPLATES}
SPECIAL = {"EMAIL": EMAIL_SPECIAL, "ADDRESS": ADDRESS_SPECIAL}

ALLOWED_EMAIL = re.compile(r"@(?:[a-z0-9-]+\.)*example\.(?:com|org|net)$", re.I)
PHONE_OK = [re.compile(p) for p in (r"555\D{0,2}01\d\d", r"7946\D?0\d\d\d", r"5550\D?\d{4}")]


def _carriers():
    splits = load_splits()
    recs = [r for r in read_jsonl(DATA / "banking_records.jsonl") if splits.get(r["record_id"]) == "train"]
    random.Random(SEED).shuffle(recs)
    return recs


def _compose(sentence_tpl, value, carrier_text, placement):
    """Return (text, start, end) with the value located inside the composed text."""
    pre, post = sentence_tpl.split("{v}")
    sentence = pre + value + post
    if placement == "prefix":
        text, offset = sentence + " " + carrier_text, 0
    else:
        text, offset = carrier_text + " " + sentence, len(carrier_text) + 1
    start = offset + len(pre)
    return text, start, start + len(value)


def build():
    carriers = iter(_carriers())
    items = []
    for pii_type, groups in FORMAT_TYPES:
        for tag, values in groups.items():
            tpls = SPECIAL.get(pii_type, {}).get(tag, TEMPLATES[pii_type])
            for k, value in enumerate(values):
                tpl, placement = tpls[k % len(tpls)]
                car = next(carriers)
                text, s, e = _compose(tpl, value, norm(car["text"]), placement)
                label = value
                if tag == "title_name":  # label the surname only
                    title, label = value.split(" ", 1)
                    s += len(title) + 1
                items.append({"text": text, "spans": [{"type": pii_type, "start": s, "end": e, "value": label}],
                              "category": pii_type, "template_id": tpl, "format_tag": tag,
                              "carrier_record_id": car["record_id"]})
    for tag, sentences in NEGATIVES.items():
        for k, sentence in enumerate(sentences):
            for placement in ("prefix", "suffix"):
                car = next(carriers)
                c = norm(car["text"])
                text = sentence + " " + c if placement == "prefix" else c + " " + sentence
                items.append({"text": text, "spans": [], "category": "negative", "template_id": sentence,
                              "format_tag": tag, "carrier_record_id": car["record_id"]})
    # stratified 50/50 dev / heldout split by (category, format_tag)
    rng = random.Random(SEED)
    groups = collections.defaultdict(list)
    for it in items:
        groups[(it["category"], it["format_tag"])].append(it)
    for key in sorted(groups):
        g = groups[key]
        rng.shuffle(g)
        for j, it in enumerate(g):
            it["subset"] = "challenge_dev" if j % 2 == 0 else "challenge_heldout"
    for n, it in enumerate(items):
        it["id"] = f"CH_{n:04d}"
    return [{k: it[k] for k in ("id", "subset", "text", "spans", "category", "template_id", "format_tag",
                                "carrier_record_id")} for it in items]


def validate(items):
    texts = set()
    for it in items:
        assert it["text"] not in texts, it["id"]
        texts.add(it["text"])
        for sp in it["spans"]:
            assert it["text"][sp["start"]:sp["end"]] == sp["value"], it["id"]
            if sp["type"] == "EMAIL":
                assert ALLOWED_EMAIL.search(sp["value"]), sp["value"]
            if sp["type"] == "PHONE":
                assert any(p.search(sp["value"]) for p in PHONE_OK), sp["value"]
    cats = collections.Counter(it["category"] for it in items)
    assert all(cats[t] >= 60 for t in ("PERSON", "EMAIL", "PHONE", "ADDRESS")) and cats["negative"] >= 60, cats
    tags = collections.Counter((it["category"], it["format_tag"]) for it in items)
    assert min(tags.values()) >= 8, tags
    return cats, tags


def main():
    items = build()
    cats, tags = validate(items)
    write_jsonl(DATA_M3 / "challenge_set.jsonl", items)
    print("items:", len(items), dict(cats))
    print("subsets:", dict(collections.Counter(it["subset"] for it in items)))
    for (c, t), n in sorted(tags.items()):
        print(f"  {c:8s} {t:18s} {n}")


if __name__ == "__main__":
    main()
