from PIL import Image, ImageDraw, ImageFont
import numpy as np

def strong_crop_white(path, bg_threshold=250):
    img = Image.open(path).convert("RGB")
    arr = np.array(img)

    mask = np.sum(arr < bg_threshold, axis=2) > 0
    coords = np.argwhere(mask)

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1

    return img.crop((x0, y0, x1, y1))

homo = strong_crop_white("homo.png")
lumo = strong_crop_white("lumo.png")

target_h = max(homo.height, lumo.height)

def resize_to_height(img, h):
    ratio = h / img.height
    return img.resize((int(img.width * ratio), h), Image.LANCZOS)

homo = resize_to_height(homo, target_h)
lumo = resize_to_height(lumo, target_h)

margin_top = int(target_h * 0.2)
margin_bottom = int(target_h * 0.2)
margin_left = int(target_h * 0.12)
margin_right = int(target_h * 0.12)
panel_gap = int(target_h * 0.25)

canvas_w = margin_left + homo.width + panel_gap + lumo.width + margin_right
canvas_h = margin_top + target_h + margin_bottom

canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
homo_x = margin_left
homo_y = margin_top
canvas.paste(homo, (homo_x, homo_y))
lumo_x = margin_left + homo.width + panel_gap
lumo_y = margin_top
canvas.paste(lumo, (lumo_x, lumo_y))

draw = ImageDraw.Draw(canvas)
try:
    font_label = ImageFont.truetype("arialbd.ttf", 80)
except IOError:
    try:
        font_label = ImageFont.truetype("arial.ttf", 80)
    except IOError:
        font_label = ImageFont.load_default()

try:
    font_text = ImageFont.truetype("arial.ttf", 72)
except IOError:
    font_text = ImageFont.load_default()

label_a = "(a)"
text_a = "HOMO"

label_a_x = homo_x
label_a_y = max(20, margin_top - int(0.75 * font_label.size))

draw.text((label_a_x, label_a_y), label_a, fill=(0, 0, 0), font=font_label)

offset_a = draw.textlength(label_a + " ", font=font_label)
draw.text(
    (label_a_x + offset_a, label_a_y + font_label.size - font_text.size),
    text_a,
    fill=(0, 0, 0),
    font=font_text,
)

label_b = "(b)"
text_b = "LUMO"

label_b_x = lumo_x
label_b_y = label_a_y

draw.text((label_b_x, label_b_y), label_b, fill=(0, 0, 0), font=font_label)

offset_b = draw.textlength(label_b + " ", font=font_label)
draw.text(
    (label_b_x + offset_b, label_b_y + font_label.size - font_text.size),
    text_b,
    fill=(0, 0, 0),
    font=font_text,
)

iso_text = "Isovalue = 0.02 a.u."
try:
    font_iso = ImageFont.truetype("arial.ttf", 60)
except IOError:
    font_iso = ImageFont.load_default()

iso_width = draw.textlength(iso_text, font=font_iso)
iso_x = (canvas_w - iso_width) // 2
iso_y = margin_top + target_h + int((margin_bottom - font_iso.size) / 2)

draw.text((iso_x, iso_y), iso_text, fill=(0, 0, 0), font=font_iso)

canvas.save("homo_lumo_2panel_npj.png", dpi=(600, 600), quality=95)
canvas.save("homo_lumo_2panel_npj.tiff", dpi=(600, 600), compression="tiff_lzw")
print("Done! homo_lumo_2panel_npj.tiff")
print("Done! homo_lumo_2panel_npj.png")
