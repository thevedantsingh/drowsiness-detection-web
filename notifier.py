import requests, time, config

last_sent = 0

def send_notification(title, message, level=1):
    global last_sent
    now = time.time()
    if (now - last_sent) < config.NOTIFICATION_COOLDOWN:
        return  # Cooldown active
    last_sent = now
    
    method = config.NOTIFY_METHOD
    
    if method == "pushover":
        _pushover(title, message, level)
    elif method == "telegram":
        _telegram(title, message)
    elif method == "pushbullet":
        _pushbullet(title, message)
    else:
        print(f"[Notify] {title}: {message}")

def _pushover(title, message, level):
    priority = {1: 0, 2: 0, 3: 1}.get(level, 0)  # Level 3 = high priority
    requests.post("https://api.pushover.net/1/messages.json", data={
        "token":    config.PUSHOVER_TOKEN,
        "user":     config.PUSHOVER_USER,
        "title":    title,
        "message":  message,
        "priority": priority,
        "sound":    "siren" if level >= 3 else "pushover"
    })
    print(f"[Pushover] Sent: {title}")

def _telegram(title, message):
    text = f"🚗 *{title}*\n{message}"
    url  = f"https://api.telegram.org/bot{config.TELEGRAM_TOKEN}/sendMessage"
    requests.post(url, json={
        "chat_id":    config.TELEGRAM_CHAT_ID,
        "text":       text,
        "parse_mode": "Markdown"
    })
    print(f"[Telegram] Sent: {title}")

def _pushbullet(title, message):
    requests.post("https://api.pushbullet.com/v2/pushes",
        headers={"Access-Token": config.PUSHBULLET_TOKEN},
        json={"type": "note", "title": title, "body": message}
    )
    print(f"[Pushbullet] Sent: {title}")