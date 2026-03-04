import cv2
import xlwt


def update_center_points(data, dic_center_points):
    '''
    更新坐标
    '''
    for row in data:
        x1, y1, x2, y2, cls_name, conf, obj_id = row[:7]

        # 计算中心点坐标
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        # 更新字典
        if obj_id in dic_center_points:
            # 判断列表长度是否超过30
            if len(dic_center_points[obj_id]) >= 30:
                dic_center_points[obj_id].pop(0)
            dic_center_points[obj_id].append((center_x, center_y))
        else:
            dic_center_points[obj_id] = [(center_x, center_y)]

    return dic_center_points


def res2OCres(results):
    lst_res = []
    if results is None:
        return lst_res
    for res in results.tolist():
        box = res[:4]
        conf = res[-2]
        cls = res[-1]
        lst_res.append([cls, conf, box])

    return list(lst_res)


def result_info_format(result_info, consum_time, results, score, cls_name):
    '''
        格式组合
    '''
    # 类别
    result_info['cls_name'] = cls_name
    # 置信度
    result_info['score'] = round(score, 2)
    # time
    result_info['time'] = consum_time
    # num
    result_info['num'] = len(results)

    return result_info


def format_data(results):
    '''
    整理模型的识别结果
    '''
    lst_results = []
    for result in results:
        name_dict = result.names
        probs = result.probs.cpu().numpy()
        top1_index = probs.top1
        class_name = name_dict[top1_index]
        top1_value = round(probs.top1conf, 2)
        lst_results.append([class_name, top1_value])
    return lst_results


def writexls(DATA, path):
    wb = xlwt.Workbook()
    ws = wb.add_sheet('Data')
    for i, Data in enumerate(DATA):
        for j, data in enumerate(Data):
            ws.write(i, j, str(data))
    wb.save(path)


def writecsv(DATA, path):
    try:
        f = open(path, 'w', encoding='utf8')
        for data in DATA:
            f.write(','.join('%s' % dat for dat in data) + '\n')
        f.close()
    except Exception as e:
        print(e)


def resize_with_padding(image, target_width, target_height, padding_value):
    """
    填充原图片的四周
    """
    # 原始图像大小
    original_height, original_width = image.shape[:2]

    # 计算宽高比例
    width_ratio = target_width / original_width
    height_ratio = target_height / original_height

    # 确定调整后的图像大小和填充大小
    if width_ratio < height_ratio:
        new_width = target_width
        new_height = int(original_height * width_ratio)
        top = (target_height - new_height) // 2
        bottom = target_height - new_height - top
        left, right = 0, 0
    else:
        new_width = int(original_width * height_ratio)
        new_height = target_height
        left = (target_width - new_width) // 2
        right = target_width - new_width - left
        top, bottom = 0, 0

    # 调整图像大小并进行固定值填充
    resized_image = cv2.resize(image, (new_width, new_height))
    padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                      value=padding_value)

    return padded_image



def compute_color_for_labels(label):
    palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_text_with_red_background(image, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, thickness=2):
    # 获取文本的大小
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)

    # 计算矩形背景的大小和位置
    background_width = text_width + 10
    background_height = text_height + 10
    background_position = (position[0], position[1] - text_height - 10)

    # 在图像上绘制红色背景矩形
    cv2.rectangle(image, background_position, (background_position[0] + background_width, background_position[1] + background_height), (0, 0, 255), cv2.FILLED)

    # 计算文本的居中位置
    text_x = background_position[0] + int((background_width - text_width) / 2)
    text_y = background_position[1] + int((background_height + text_height) / 2)

    # 在图像上绘制文本
    cv2.putText(image, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return image


def draw_info(frame, results):
    for i, txt in enumerate(results):
        # cv2.putText(frame, txt[0], (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        frame = draw_text_with_red_background(frame, txt[0], (0, 30), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, thickness=2)

    return frame

