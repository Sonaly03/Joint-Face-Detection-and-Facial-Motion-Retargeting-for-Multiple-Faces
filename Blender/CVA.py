
import bpy
from imutils import face_utils
import dlib
import cv2
import time
import numpy
from bpy.props import FloatProperty
    
class CV(bpy.types.Operator):

    bl_idname = "mw.opencv_operator"
    bl_label = "CV Project"
    #Path for Landmark points location 
    l = "/Users/Rohit/Desktop/Python/CV/shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(l)

    _timer = None
    _cap  = None
    
    width = 850
    height = 650

    stop :bpy.props.BoolProperty()

    #3D Modal Points  
    mpoints = numpy.array([
                                (0.0, 0.0, 0.0),  #nose           
                                (0.0, -330.5, -65.0),  #chin      
                                (-225.0, 170.0, -135.0),   #left eye left corner	  
                                (225.5, 170.0, -135.0),   #right eye right corner   
                                (-150.5, -150.0, -125.0),    #left mouth corner
                                (150.0, -150.5, -125.0)    #right mouth corner  
                            ], dtype = numpy.float32)
    #Camara Matrix	
    cmatrix = numpy.array(
                            [[height, 0.0, width/2],
                            [0.0, height, height/2],
                            [0.0, 0.0, 1.0]], dtype = numpy.float32
                            )
    #Maintains a moving average of the length provided
    def s_v(m, n, length, v):
        if not hasattr(m, 'smooth'):
            m.smooth = {}
        if not n in m.smooth:
            m.smooth[n] = numpy.array([v])
        else:
            m.smooth[n] = numpy.insert(arr=m.smooth[n], obj=0, values=v)
            if m.smooth[n].size > length:
                m.smooth[n] = numpy.delete(m.smooth[n], m.smooth[n].size-1, 0)
        sum = 0
        for val in m.smooth[n]:
            sum += val
        return sum / m.smooth[n].size
    
    def exit(m, c):
        mw = c.window_manager #open window manager and UI
        mw.event_timer_remove(m._timer) #add timer to the window
        cv2.destroyAllWindows()
        m._cap.release()
        m._cap = None


    def get_range(m, n, v):
        if not hasattr(m, 'range'):
            m.range = {}
        if not n in m.range:
            m.range[n] = numpy.array([v, v])
        else:
            m.range[n] = numpy.array([min(v, m.range[n][0]), max(v, m.range[n][1])] )
        v_r = m.range[n][1] - m.range[n][0]
        if v_r != 0:
            return (v - m.range[n][0]) / v_r
        else:
            return 0
        
    def execute(m, c):
        bpy.app.handlers.frame_change_pre.append(m.endplay)

        mw = c.window_manager
        m._timer = mw.event_timer_add(0.03, window=c.window)
        mw.modal_handler_add(m)
        return {'RUNNING_MODAL'}
    

    def modal(m, c, e):

        if (e.type in {'RIGHTMOUSE', 'ESC'}) or m.stop == True:
            m.exit(c) #exits
            return {'CANCELLED'}

        if e.type == 'TIMER':
            m.camera()
            _, image = m._cap.read()
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            rects = m.detector(gray, 0)
	#Find landmark for each detected face
            for (i, rect) in enumerate(rects):
                s = m.predictor(gray, rect)
                s = face_utils.shape_to_np(s)
          	#2D Points
                image_points = numpy.array([s[30],     #nose tip
                                            s[8],      #chin
                                            s[36],     #left eye left corner
                                            s[45],     #right eye right corner
                                            s[48],     #left mouth corner
                                            s[54]      #right mouth cormer
                                        ], dtype = numpy.float32)
             
                coeffs = numpy.zeros((4,1)) #return new array of given shape and type.
             	#Modal co-ordinate system to 3D conversion
                if hasattr(m, 'rotation_vector'):
                    (success, m.rotation_vector, m.translation_vector) = cv2.solvePnP(m.mpoints, 
                        image_points, m.cmatrix, coeffs, flags=cv2.SOLVEPNP_ITERATIVE, 
                        rvec=m.rotation_vector, tvec=m.translation_vector, 
                        useExtrinsicGuess=True)
                else:
                    (success, m.rotation_vector, m.translation_vector) = cv2.solvePnP(m.mpoints, 
                        image_points, m.cmatrix, coeffs, flags=cv2.SOLVEPNP_ITERATIVE, 
                        useExtrinsicGuess=False)
             
                if not hasattr(m, 'first_angle'):
                    m.first_angle = numpy.copy(m.rotation_vector)
             	#Using blender charecter pre trained model named "Vincent"
                b = bpy.data.objects["RIG-Vincent"].pose.bones
                #to move the head position 
                b["head_fk"].rotation_euler[0] = m.s_v("h_x", 3, (m.rotation_vector[0] - m.first_angle[0])) / 1   
                b["head_fk"].rotation_euler[2] = m.s_v("h_y", 3, -(m.rotation_vector[1] - m.first_angle[1])) / 1.5  
                b["head_fk"].rotation_euler[1] = m.s_v("h_z", 3, (m.rotation_vector[2] - m.first_angle[2])) / 1.3   
                
                b["mouth_ctrl"].location[2] = m.s_v("m_h", 2, -m.get_range("mouth_height", numpy.linalg.norm(s[62] - s[66])) * 0.06 )
                b["mouth_ctrl"].location[0] = m.s_v("m_w", 2, (m.get_range("mouth_width", numpy.linalg.norm(s[54] - s[48])) - 0.5) * -0.04)
                b["brow_ctrl_L"].location[2] = m.s_v("b_l", 3, (m.get_range("brow_left", numpy.linalg.norm(s[19] - s[27])) -0.5) * 0.04)
                b["brow_ctrl_R"].location[2] = m.s_v("b_r", 3, (m.get_range("brow_right", numpy.linalg.norm(s[24] - s[27])) -0.5) * 0.04)
                #code for smooth transition
                b["head_fk"].keyframe_insert(data_path="rotation_euler", index=-1)
                b["mouth_ctrl"].keyframe_insert(data_path="location", index=-1)
                b["brow_ctrl_L"].keyframe_insert(data_path="location", index=2)
                b["brow_ctrl_R"].keyframe_insert(data_path="location", index=2)
         	#draw circle using cv2 function
                for (x, y) in s:
                    cv2.circle(image, (x, y), 2, (0, 255, 255), -1)
                 
            cv2.imshow("Output", image)
            cv2.waitKey(1)

        return {'PASS_THROUGH'}

    
    def endplay(m, o):
        print(format(o.frame_current) + " / " + format(o.frame_end))
        if o.frame_current == o.frame_end:
            bpy.ops.screen.animation_cancel(restore_frame=False)
    
    def camera(m):
        if m._cap == None:
            m._cap = cv2.VideoCapture(0)
            m._cap.set(cv2.CAP_PROP_FRAME_WIDTH, m.width)
            m._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, m.height)
            m._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) #no of frames stored in buffer
            time.sleep(0.5)

def to_register():
    bpy.utils.register_class(CV)

def to_unregister():
    bpy.utils.unregister_class(CV)

if __name__ == "__main__":
    to_register()

     #test call
bpy.ops.mw.opencv_operator()
