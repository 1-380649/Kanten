from pynput.keyboard import Listener

def on_press(key):
    try:
        print(f"Alphanumeric key {key.char} pressed")
    except AttributeError:
        print(f"Special key {key} pressed")

# Sammelt Ereignisse, bis der Block verlassen wird
with Listener(on_press=on_press) as listener:
    listener.join()