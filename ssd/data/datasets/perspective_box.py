import cv2
import math
import numpy as np

import torch.utils.data

from PIL import Image, ImageDraw 

from ssd.structures.container import Container

class BoundingBoxSampler:
    """ 
    Generator of rectangular bounding boxes
    
    Creates rectangle samples from a normal distribution 
    """
    def __init__( self, 
                 width = 320, 
                 height = 320, 
                 width_reach = 0.65,
                 height_reach = 0.65,
                 mean_width = 125,
                 min_width = 50,
                 max_width = 200,
                 mean_height = 75,
                 min_height = 25,
                 max_height = 125
        ):
        
        # set image properties
        self.width = width 
        self.height = height
        self.width_reach = width_reach
        self.height_reach = height_reach
        
        # set width properties
        self.mean_width = mean_width
        self.n_width = mean_width - min_width
        self.p_width = max_width - mean_width
        
        # set height properties
        self.mean_height = mean_height
        self.n_height = mean_height - min_height 
        self.p_height = max_height - mean_height
    
    # get N random samples
    def sample( self, count = 1 ):   
        # uniform cx, cy
        box_x = ( np.random.random( count ) - 0.5 ) * self.width_reach + 0.5
        box_y = ( np.random.random( count ) - 0.5 ) * self.height_reach + 0.5
        
        # box width
        box_w_spread = np.clip( np.random.normal( 0, 0.225, size = count ), -1.00, 1.00 )
        box_w_nmap = ( box_w_spread < 0.00 )
        
        box_w = np.zeros( count, dtype = np.float )
        np.putmask( box_w, box_w_nmap, self.mean_width + ( box_w_spread * self.n_width ) )
        np.putmask( box_w, np.logical_not( box_w_nmap ), self.mean_width + ( box_w_spread * self.p_width ) )
        
        # aspect ratio
        box_h_spread = np.clip( np.random.normal( 0, 0.225, size = count ), -1.00, 1.00 )
        box_h_nmap = ( box_h_spread < 0.00 )
        
        box_h = np.zeros( count, dtype = np.float )
        np.putmask( box_h, box_h_nmap, self.mean_height + ( box_h_spread * self.n_height ) )
        np.putmask( box_h, np.logical_not( box_h_nmap ), self.mean_height + ( box_h_spread * self.p_height ) )
        
        return np.vstack( ( box_x, box_y, ( box_w / self.width ), ( box_h / self.height ) ) ).T
        
class BoxTranslate:
    def __init__( self, x = 0, y = 0, z = 0 ):
        self.transform = np.eye( 4 )
        self.transform[ 0, 3 ] = x
        self.transform[ 1, 3 ] = y
        self.transform[ 2, 3 ] = z
        
    def __call__( self, points ):
        result = np.matmul( points, self.transform.T )
        return result / result[ :, 3 ][ :, np.newaxis ]
    
class BoxRotate:
    def __init__( self, rx = 0, ry = 0, rz = 0 ):
        self.transform = self.rx_transform( rx )
        self.transform = np.matmul( self.transform, self.ry_transform( ry ) )
        self.transform = np.matmul( self.transform, self.rz_transform( rz ) )   
    
    def rx_transform( self, rx ):
        transform = np.eye( 4, dtype = np.float )
        transform[ 1, 1 ] = np.cos( rx )
        transform[ 2, 2 ] = transform[ 1, 1 ]
        transform[ 2, 1 ] = np.sin( rx )
        transform[ 1, 2 ] = - transform[ 2, 1 ]
        return transform
    
    def ry_transform( self, ry ):
        transform = np.eye( 4, dtype = np.float )
        transform[ 0, 0 ] = np.cos( ry )
        transform[ 2, 2 ] = transform[ 0, 0 ]
        transform[ 0, 2 ] = np.sin( ry )
        transform[ 2, 0 ] = - transform[ 0, 2 ]
        return transform
    
    def rz_transform( self, rz ):
        transform = np.eye( 4, dtype = np.float )
        transform[ 0, 0 ] = np.cos( rz )
        transform[ 1, 1 ] = transform[ 0, 0 ]
        transform[ 1, 0 ] = np.sin( rz )
        transform[ 0, 1 ] = - transform[ 1, 0 ]
        return transform
    
    def __call__( self, points ):
        result = np.matmul( points, self.transform.T )
        return result / result[ :, 3 ][ :, np.newaxis ]

class CameraProjection:
    def __init__( self, width, height, distance = 10000 ):
        self.transform = BoxTranslate( 0, 0, distance ).transform
        self.distance = distance
    
    def __call__( self, points ):
        result = np.matmul( points, self.transform.T )
        result = result / result[ :, 3 ][ :, np.newaxis ]
        return result / result[ :, 2 ][ :, np.newaxis ] * self.distance


def Box3DPoints( width, height ):
    points = np.zeros( ( 4, 4 ), dtype = np.float )
    w = width / 2
    h = height / 2
    
    points[ 0, 0:2 ] = [ - w, - h ]
    points[ 1, 0:2 ] = [ - w, h ]
    points[ 2, 0:2 ] = [ w, h ]
    points[ 3, 0:2 ] = [ w, -h ]
    
    points[ :, 3 ] = 1
    
    return points


class RotatedBoundingBoxSampler( BoundingBoxSampler ):
    """ 
    Sampler for rotated bounding boxes
    """
    def __init__( self, x_range = 45, y_range = 45, z_range = 45, width = 320, height = 320, **kwargs ):
        # init sampler
        super( RotatedBoundingBoxSampler, self ).__init__( width, height, **kwargs )
        self.width = width
        self.height = height
        self.x_range = x_range / 180 * np.pi
        self.y_range = y_range / 180 * np.pi
        self.z_range = z_range / 180 * np.pi
        
    def __call__( self, count = 1 ):
        boxes = self.sample( count )
        
        # calculate WxH
        boxes[ :, 0 ] *= self.width
        boxes[ :, 1 ] *= self.height
        # calculate XY
        boxes[ :, 2 ] *= self.width
        boxes[ :, 3 ] *= self.height
        
        # angles
        z_alpha = ( np.random.random( count ) - 0.5 ) / 0.5 * self.z_range
        x_alpha = ( np.random.random( count ) - 0.5 ) / 0.5 * self.x_range
        y_alpha = ( np.random.random( count ) - 0.5 ) / 0.5 * self.y_range
        
        # camera projection
        project_2d = CameraProjection( self.width, self.height )
        
        points = np.zeros( ( count, 4, 2 ), dtype = np.float )
        for i in range( count ):
            box_points = Box3DPoints( boxes[ i, 2 ], boxes[ i, 3 ] )
            box_points = BoxRotate( x_alpha[ i ], y_alpha[ i ], z_alpha[ i ] )( box_points )
            box_points = BoxTranslate( boxes[ i, 0 ], boxes[ i, 1 ] )( box_points )
            
            points[ i, :, : ] = project_2d( box_points )[ :, 0:2 ]
        
        points = np.array( points, dtype = np.int )
        
        contours = self.RotatedBoxContour( points )
        boxes = self.InternalMinMaxBoxes( contours )
        
        return ( contours, boxes )
    
    def RotatedBoxContour( self, boxes ):
        contours = []
        for i in range( len( boxes ) ):
            contours.append( [ [ boxes[ i ][ k % 4, 0 ], boxes[ i ][ k % 4, 1 ] ] for k in range( 5 ) ] )
        return np.array( contours )
    
    def EnclosingBoundingBox( self, boxes ):
        boxes = np.array( boxes )
        min_x = np.clip( np.min( boxes[ :, :, 0 ], axis = 1 ), 0, self.width )
        max_x = np.clip( np.max( boxes[ :, :, 0 ], axis = 1 ), 0, self.width )
        min_y = np.clip( np.min( boxes[ :, :, 1 ], axis = 1 ), 0, self.height )
        max_y = np.clip( np.max( boxes[ :, :, 1 ], axis = 1 ), 0, self.height )

        return np.array( [ min_x, min_y, max_x, max_y ] ).T

    def InternalMinMaxBoxes( self, boxes ):
        # clip contour points to image dimensions
        boxes[ :, :, 0 ] = np.clip( boxes[ :, :, 0 ], 0, self.width )
        boxes[ :, :, 1 ] = np.clip( boxes[ :, :, 1 ], 0, self.height )
        
        # create candidate arrays
        box_a_pt = np.hstack( ( boxes[ :, 0, : ], boxes[ :, 2, : ] ) )
        box_b_pt = np.hstack( ( boxes[ :, 1, : ], boxes[ :, 3, : ] ) )
        
        # box areas
        box_a_areas = np.abs( box_a_pt[ :, 2 ] - box_a_pt[ :, 0 ] ) * np.abs( box_a_pt[ :, 3 ] - box_a_pt[ :, 1 ] )
        box_b_areas = np.abs( box_b_pt[ :, 2 ] - box_b_pt[ :, 0 ] ) * np.abs( box_b_pt[ :, 3 ] - box_b_pt[ :, 1 ] )
        
        # select small boxes
        boxes_sm = np.zeros_like( box_a_pt )
        boxes_lg = np.zeros_like( box_b_pt )

        # compare areas
        box_select_sm = box_a_areas < box_b_areas
        box_select_lg = np.logical_not( box_select_sm )

        # create arrays
        boxes_sm[ box_select_sm ] = box_a_pt[ box_select_sm ]
        boxes_lg[ box_select_sm ] = box_b_pt[ box_select_sm ]
        boxes_sm[ box_select_lg ] = box_b_pt[ box_select_lg ]
        boxes_lg[ box_select_lg ] = box_a_pt[ box_select_lg ]
        
        def enclosingBoxes( bx ):
            min_x = np.min( np.vstack( ( bx[ :, 0 ], bx[ :, 2 ] ) ).T, axis = 1 ) 
            max_x = np.max( np.vstack( ( bx[ :, 0 ], bx[ :, 2 ] ) ).T, axis = 1 )
            min_y = np.min( np.vstack( ( bx[ :, 1 ], bx[ :, 3 ] ) ).T, axis = 1 )
            max_y = np.max( np.vstack( ( bx[ :, 1 ], bx[ :, 3 ] ) ).T, axis = 1 )
            return np.array( [ min_x, min_y, max_x, max_y ] ).T
        
        def normalizeBoxes( bx ):
            bx = np.array( bx, dtype = np.float ) 
            bx[ :, 0 ] = bx[ :, 0 ] / self.width
            bx[ :, 2 ] = bx[ :, 2 ] / self.width
            bx[ :, 1 ] = bx[ :, 1 ] / self.height
            bx[ :, 3 ] = bx[ :, 3 ] / self.height
            return bx
        
        return [ 
            normalizeBoxes( enclosingBoxes( boxes_sm ) ), 
            normalizeBoxes( enclosingBoxes( boxes_lg ) )
        ]        
    
    def NormalizedEnclosingBoundingBox( self, boxes ):
        boxes = np.array( boxes, dtype = np.float ) 
        boxes[ :, 0 ] = boxes[ :, 0 ] / self.width
        boxes[ :, 2 ] = boxes[ :, 2 ] / self.width
        boxes[ :, 1 ] = boxes[ :, 1 ] / self.height
        boxes[ :, 3 ] = boxes[ :, 3 ] / self.height
        return boxes
    
    def DrawRotatedRectangle( self, contour ):
        img = np.zeros( ( self.height, self.width, 3 ), dtype = np.uint8 )
        cv2.drawContours( img, [ contour ], 0, ( 255, 255, 255 ), -1 )
        return img
    
    def DrawBoundingBox( self, img, box, color = ( 255, 0, 0 ) ):
        cv2.rectangle( img, 
                      ( int( box[ 0 ] * self.width ), int( box[ 1 ] * self.height ) ), 
                      ( int( box[ 2 ] * self.width ), int( box[ 3 ] * self.height ) ), 
                      color, 2 )
        return img
    



class PerspectiveBoxes( torch.utils.data.Dataset ):
    
    class_names = [ 'background', 'label_sm, label_lg' ]

    def __init__( self, examples = 10240, transform = None, target_transform = None ):
        # as you would do normally
        self.generator = RotatedBoundingBoxSampler()
        self.length = examples
        self.contours, self.boxes = self.generator( examples )
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        # load the image as a PIL Image
        image = self.generator.DrawRotatedRectangle( self.contours[ index ] )
        
        # load the bounding boxes in x1, y1, x2, y2 order.
        boxes = np.array( [ self.boxes[ 0 ][ index ], self.boxes[ 1 ][ index ] ], dtype = np.float32 )
        # and labels
        labels = np.array( [ [ 1, 2 ] ], dtype=np.int64)

        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        targets = Container(
            boxes=boxes,
            labels=labels,
        )
        # return the image, the targets and the index in your dataset
        return image, targets, index

    def __len__( self ):
        return self.length

    def _get_annotation( self, index ):
        return ( np.array( [ self.boxes[ index ] ], dtype=np.float32 ),
                np.array( [ 1 ], dtype=np.int64 ),
                np.array( [ 0 ], dtype=np.uint8 ) )

    def get_annotation( self, index ):
        return ( 'img_' + str( index ), self._get_annotation( index ) )

    def get_img_info( self, index ):
        return dict( width = self.generator.width, height = self.generator.height )


