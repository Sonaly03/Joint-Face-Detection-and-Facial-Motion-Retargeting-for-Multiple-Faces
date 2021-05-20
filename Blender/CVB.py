import bpy


class CV_Project(bpy.types.WorkSpaceTool):

    bl_label = "CV Project"
    bl_space_type = 'VIEW_3D'
    bl_context_mode='OBJECT'
        
    
    bl_idname = "ui_plusr0rf8.opencv"
    bl_options = {'REGISTER'}

    bl_icon = "ops.generic.select_circle"

        
    def draw(context, layout, tool):

        store_row = layout.row()
        store_op = store_row.operator("wm.opencv_operator", text="Capture", icon="OUTLINER_OB_CAMERA")
        

def to_register():
    
    bpy.utils.register_tool(CV_Project, separator=True, group=True)

def to_unregister():

    bpy.utils.unregister_tool(CV_Project)

if __name__ == "__main__":
    to_register()