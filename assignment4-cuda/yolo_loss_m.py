import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def compute_iou(box1, box2):
    """Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].
    Args:
      box1: (tensor) bounding boxes, sized [N,4].
      box2: (tensor) bounding boxes, sized [M,4].
    Return:
      (tensor) iou, sized [N,M].
    """
    N = box1.size(0)
    M = box2.size(0)

    lt = torch.max(
        box1[:, :2].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
        box2[:, :2].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
    )

    rb = torch.min(
        box1[:, 2:].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
        box2[:, 2:].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
    )

    wh = rb - lt  # [N,M,2]
    wh[wh < 0] = 0  # clip at 0
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N,]
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M,]
    area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
    area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

    iou = inter / (area1 + area2 - inter)
    return iou


class YoloLoss(nn.Module):
    def __init__(self, S, B, l_coord, l_noobj):
        super(YoloLoss, self).__init__()
        self.S = S
        self.B = B
        self.l_coord = l_coord
        self.l_noobj = l_noobj

    def xywh2xyxy(self, boxes):
        """
        Parameters:
        boxes: (N,4) representing by x,y,w,h

        Returns:
        boxes: (N,4) representing by x1,y1,x2,y2

        if for a Box b the coordinates are represented by [x, y, w, h] then
        x1, y1 = x/S - 0.5*w, y/S - 0.5*h ; x2,y2 = x/S + 0.5*w, y/S + 0.5*h
        Note: Over here initially x, y are the center of the box and w,h are width and height.
        """
        ### CODE ###
        # Your code here
        x1= boxes[:, 0]/self.S - 0.5*boxes[:, 2]
        y1 = boxes[:, 1]/self.S - 0.5*boxes[:, 3]
        x2= boxes[:, 0]/self.S + 0.5*boxes[:, 2]
        y2 = boxes[:, 1]/self.S + 0.5*boxes[:, 3]
        boxes = torch.stack((x1, y1, x2, y2), axis=1)
        return boxes

    def find_best_iou_boxes(self, pred_box_list, box_target):
        """
        Parameters:
        box_pred_list : [(tensor) size (-1, 5) ...]
        box_target : (tensor)  size (-1, 4)

        Returns:
        best_iou: (tensor) size (-1, 1)
        best_boxes : (tensor) size (-1, 5), containing the boxes which give the best iou among the two (self.B) predictions

        Hints:
        1) Find the iou's of each of the 2 bounding boxes of each grid cell of each image.
        2) For finding iou's use the compute_iou function
        3) use xywh2xyxy to convert bbox format if necessary,
        Note: Over here initially x, y are the center of the box and w,h are width and height.
        We perform this transformation to convert the correct coordinates into bounding box coordinates.
        """

        ### CODE ###
        # Your code here
        N = box_target.shape[0]
        best_iou = torch.zeros(N)
        best_boxes = torch.zeros(N, 5)
        target_box_xyxy = self.xywh2xyxy(box_target)
        
        
        ious = torch.zeros(N, self.B)#.to(device)
        for box in range(len(pred_box_list)) :
            iou = compute_iou(pred_box_list[box][:, :4], target_box_xyxy)
            iou = iou.diag()
            ious[:, box]= iou
        for idx in range(N):
           best_iou[idx] = torch.max(ious[idx])
           best_boxes[idx] = pred_box_list[torch.argmax(ious[idx])][idx]
        
        best_iou= best_iou.unsqueeze(1).detach()
        return best_iou, best_boxes

    def get_class_prediction_loss(self, classes_pred, classes_target, has_object_map):
        """
        Parameters:
        classes_pred : (tensor) size (batch_size, S, S, 20)
        classes_target : (tensor) size (batch_size, S, S, 20)
        has_object_map: (tensor) size (batch_size, S, S)

        Returns:
        class_loss : scalar
        """
        ### CODE ###
        # Your code here
        sum_of_squared_errors = torch.sum(torch.square(classes_pred - classes_target), dim = -1)
        # print("sum_of_squared_errors dims ", sum_of_squared_errors.shape)
        loss = torch.sum(torch.multiply(has_object_map, sum_of_squared_errors))/classes_pred.shape[0]
        return loss

    def get_no_object_loss(self, pred_boxes_list, has_object_map):
        """
        Parameters:
        pred_boxes_list: (list) [(tensor) size (N, S, S, 5)  for B pred_boxes]
        has_object_map: (tensor) size (N, S, S)

        Returns:
        loss : scalar

        Hints:
        1) Only compute loss for cell which doesn't contain object
        2) compute loss for all predictions in the pred_boxes_list list
        3) You can assume the ground truth confidence of non-object cells is 0
        """
        ### CODE ###
        # Your code here
        # print(has_object_map.dtype)
        has_no_obj_map = (has_object_map==0)
        loss = 0.0
        for i in range(len(pred_boxes_list)):
          loss += torch.sum(torch.square(pred_boxes_list[i][:, :, :, 4][has_no_obj_map]))/pred_boxes_list[i].shape[0]
        loss *= self.l_noobj
        # print("no obj loss ", loss)
        return loss

    def get_contain_conf_loss(self, box_pred_conf, box_target_conf):
        """
        Parameters:
        box_pred_conf : (tensor) size (-1,1)
        box_target_conf: (tensor) size (-1,1)

        Returns:
        contain_loss : scalar

        Hints:
        The box_target_conf should be treated as ground truth, i.e., no gradient

        """
        ### CODE
        # your code here
        loss = F.mse_loss(box_pred_conf, box_target_conf.unsqueeze(-1), reduction='sum')
        return loss

    def get_regression_loss(self, box_pred_response, box_target_response):
        """
        Parameters:
        box_pred_response : (tensor) size (-1, 4)
        box_target_response : (tensor) size (-1, 4)
        Note : -1 corresponds to ravels the tensor into the dimension specified
        See : https://pytorch.org/docs/stable/tensors.html#torch.Tensor.view_as

        Returns:
        reg_loss : scalar

        """
        ### CODE
        # your code here
        box_pred_response = box_pred_response.to('cuda')
        box_target_response = box_target_response.to('cuda')
        box_pred_response[:, 2:] = torch.sqrt(box_pred_response[:, 2:])
        box_target_response[:, 2:] = torch.sqrt(box_target_response[:, 2:])
        reg_loss = F.mse_loss(box_pred_response[:, :4], box_target_response, reduction='sum')
        reg_loss *= self.l_coord
        return reg_loss

    def forward(self, pred_tensor, target_boxes, target_cls, has_object_map):
        """
        pred_tensor: (tensor) size(N,S,S,Bx5+20=30) N:batch_size
                      where B - number of bounding boxes this grid cell is a part of = 2
                            5 - number of bounding box values corresponding to [x, y, w, h, c]
                                where x - x_coord, y - y_coord, w - width, h - height, c - confidence of having an object
                            20 - number of classes

        target_boxes: (tensor) size (N, S, S, 4): the ground truth bounding boxes
        target_cls: (tensor) size (N, S, S, 20): the ground truth class
        has_object_map: (tensor, bool) size (N, S, S): the ground truth for whether each cell contains an object (True/False)

        Returns:
        loss_dict (dict): with key value stored for total_loss, reg_loss, containing_obj_loss, no_obj_loss and cls_loss
        """
        N = pred_tensor.size(0)
        total_loss = 0.0

        # split the pred tensor from an entity to separate tensors:
        # -- pred_boxes_list: a list containing all bbox prediction (list) [(tensor) size (N, S, S, 5)  for B pred_boxes]
        # -- pred_cls (containing all classification prediction)
        pred_boxes_list = []
        for b in range(self.B):
          pred_boxes_list.append(pred_tensor[:, :, :, b*5:(b+1)*5])
        
        classes_pred = pred_tensor[:, :, :, self.B*5:]

        # compcute classification loss
        classification_loss = self.get_class_prediction_loss(classes_pred, target_cls, has_object_map)
        # print("classification_loss : ", classification_loss)
        # compute no-object loss
        no_obj_loss = self.get_no_object_loss(pred_boxes_list, has_object_map)

        # Re-shape boxes in pred_boxes_list and target_boxes to meet the following desires
        # 1) only keep having-object cells
        # 2) vectorize all dimensions except for the last one for faster computation
        for b in range(self.B):
          pred_boxes_list[b] = pred_boxes_list[b][has_object_map]
        
        target_boxes_with_obj = target_boxes[has_object_map]
        # find the best boxes among the 2 (or self.B) predicted boxes and the corresponding iou
        best_iou, best_boxes = self.find_best_iou_boxes(pred_boxes_list, target_boxes_with_obj)

        # compute regression loss between the found best bbox and GT bbox for all the cell containing objects
        regression_loss = self.get_regression_loss(best_boxes, target_boxes_with_obj)
        # print("regression_loss : ", regression_loss)
        # compute contain_object_loss
        contain_object_loss = self.get_contain_conf_loss(best_boxes[:,4].unsqueeze(-1), torch.ones(best_boxes.shape[0]))
        # print("contain_object_loss : ", contain_object_loss)
        # compute final loss
        total_loss = (classification_loss + no_obj_loss + regression_loss + contain_object_loss)/N
        # print("total loss : ", total_loss)
        # construct return loss_dict
        loss_dict = dict(
            total_loss=total_loss,
            reg_loss=regression_loss,
            containing_obj_loss=contain_object_loss,
            no_obj_loss=no_obj_loss,
            cls_loss=classification_loss,
        )
        return loss_dict
