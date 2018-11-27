import glfw
from mjlib import mjlib
from ctypes import pointer, byref
import ctypes
import mjcore
import os
import numpy as np
from threading import Lock
from mjconstants import *
import mjextra

mjCAT_ALL = 7

class MjViewer(object):

    def __init__(self):
        self.last_render_time = 0
        self.objects = mjcore.MJVOBJECTS()
        self.cam = mjcore.MJVCAMERA()
        self.vopt = mjcore.MJVOPTION()
        self.ropt = mjcore.MJROPTION()
        self.con = mjcore.MJRCONTEXT()
        self.running = False
        self.speedtype = 1
        self.window = None
        self.model = None
        self.gui_lock = Lock()

        self.last_button = 0
        self.last_click_time = 0
        self.button_left_pressed = False
        self.button_middle_pressed = False
        self.button_right_pressed = False
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        self.frames = []

    def set_model(self, model):
        self.model = model
        if model:
            self.data = model.data
        else:
            self.data = None
        if self.running:
            if model:
                mjlib.mjr_makeContext(model.ptr, byref(self.con), 150)
            else:
                mjlib.mjr_makeContext(None, byref(self.con), 150)
            self.render()
        if model:
            self.autoscale()

    def autoscale(self):
        self.cam.lookat[0] = self.model.stat.center[0]
        self.cam.lookat[1] = self.model.stat.center[1]
        self.cam.lookat[2] = self.model.stat.center[2]
        self.cam.distance = 1.0 * self.model.stat.extent
        self.cam.camid = -1
        self.cam.trackbodyid = -1
        if self.window:
            width, height = glfw.get_framebuffer_size(self.window)
            mjlib.mjv_updateCameraPose(byref(self.cam), width*1.0/height)

    def get_rect(self):
        rect = mjcore.MJRRECT(0, 0, 0, 0)
        rect.width, rect.height = glfw.get_framebuffer_size(self.window)
        return rect

    def record_frame(self, **kwargs):
        self.frames.append({ 'pos': self.model.data.qpos, 'extra': kwargs })

    def clear_frames(self):
        self.frames = []

    def render(self):
        rect = self.get_rect()
        arr = (ctypes.c_double*3)(0, 0, 0)

        mjlib.mjv_makeGeoms(self.model.ptr, self.data.ptr, byref(self.objects), byref(self.vopt), mjCAT_ALL, 0, None, None, ctypes.cast(arr, ctypes.POINTER(ctypes.c_double)))
        mjlib.mjv_makeLights(self.model.ptr, self.data.ptr, byref(self.objects))

        mjlib.mjv_setCamera(self.model.ptr, self.data.ptr, byref(self.cam))

        prev_pos = self.model.data.qpos
        prev_alpha = self.model.vis.map_.alpha

        self.model.vis.map_.alpha = 0.01
        #tmpobjects = mjcore.MJVOBJECTS()
        #mjlib.mjv_makeObjects(byref(tmpobjects), 1000)
        #for idx, frame in enumerate(self.frames):
        #    #print 'painting fixed'
        #    self.model.data.qpos = frame['pos']
        #    self.model.forward()
        #    #if idx == 0:
        #    #    mjlib.mjv_makeGeoms(self.model.ptr, self.data.ptr, byref(self.objects), byref(self.vopt), mjCAT_ALL, 0, None, None, ctypes.cast(arr, ctypes.POINTER(ctypes.c_double)))
        #    #else:
        #    mjlib.mjv_makeGeoms(self.model.ptr, self.data.ptr, byref(tmpobjects), byref(self.vopt), mjCAT_ALL, 0, None, None, ctypes.cast(arr, ctypes.POINTER(ctypes.c_double)))
        #    for i in range(tmpobjects.ngeom):
        #        alpha = frame['extra'].get('alpha', None)
        #        emission = frame['extra'].get('emission', None)
        #        #= frame['extra'].get('emission', None)
        #        geom = tmpobjects.geoms[i]
        #        if alpha is not None:
        #            geom.rgba[3] = alpha
        #        if emission is not None:
        #            geom.emission = emission
        #    mjextra.append_objects(self.objects, tmpobjects)

        self.model.vis.map_.alpha = prev_alpha
        self.model.data.qpos = prev_pos
        self.model.forward()
        mjlib.mjv_updateCameraPose(byref(self.cam), rect.width*1.0/rect.height)
        mjlib.mjr_render(0, rect, byref(self.objects), byref(self.ropt), byref(self.cam.pose), byref(self.con))

    def render_internal(self):
        if not self.data:
            return
        self.gui_lock.acquire()
        self.render()

        self.gui_lock.release()

    def start(self):
        if not glfw.init():
            return

        glfw.window_hint(glfw.SAMPLES, 4)

        # try stereo if refresh rate is at least 100Hz
        window = None
        stereo_available = False

        _, _, refresh_rate = glfw.get_video_mode(glfw.get_primary_monitor())
        if refresh_rate >= 100:
            glfw.window_hint(glfw.STEREO, 1)
            window = glfw.create_window(500, 500, "Simulate", None, None)
            if window:
                stereo_available = True

        # no stereo: try mono
        if not window:
            glfw.window_hint(glfw.STEREO, 0)
            window = glfw.create_window(500, 500, "Simulate", None, None)

        if not window:
            glfw.terminate()
            return

        self.running = True

        # Make the window's context current
        glfw.make_context_current(window)

        width, height = glfw.get_framebuffer_size(window)
        width1, height = glfw.get_window_size(window)
        self.scale = width * 1.0 / width1

        self.window = window

        mjlib.mjv_makeObjects(byref(self.objects), 1000)

        mjlib.mjv_defaultCamera(byref(self.cam))
        mjlib.mjv_defaultOption(byref(self.vopt))
        mjlib.mjr_defaultOption(byref(self.ropt))

        mjlib.mjr_defaultContext(byref(self.con))

        if self.model:
            mjlib.mjr_makeContext(self.model.ptr, byref(self.con), 150)
            self.autoscale()
        else:
            mjlib.mjr_makeContext(None, byref(self.con), 150)

        glfw.set_cursor_pos_callback(window, self.handle_mouse_move)
        glfw.set_mouse_button_callback(window, self.handle_mouse_button)
        glfw.set_scroll_callback(window, self.handle_scroll)

    def handle_mouse_move(self, window, xpos, ypos):

        # no buttons down: nothing to do
        if not self.button_left_pressed \
                and not self.button_middle_pressed \
                and not self.button_right_pressed:
            return

        # compute mouse displacement, save
        dx = int(self.scale * xpos) - self.last_mouse_x
        dy = int(self.scale * ypos) - self.last_mouse_y
        self.last_mouse_x = int(self.scale * xpos)
        self.last_mouse_y = int(self.scale * ypos)

        # require model
        if not self.model:
            return

        # get current window size
        width, height = glfw.get_framebuffer_size(self.window)

        # get shift key state
        mod_shift = glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS \
                or glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS

        # determine action based on mouse button
        action = None
        if self.button_right_pressed:
            action = MOUSE_MOVE_H if mod_shift else MOUSE_MOVE_V
        elif self.button_left_pressed:
            action = MOUSE_ROTATE_H if mod_shift else MOUSE_ROTATE_V
        else:
            action = MOUSE_ZOOM

        self.gui_lock.acquire()

        mjlib.mjv_moveCamera(action, dx, dy, byref(self.cam), width, height)

        self.gui_lock.release()


    def handle_mouse_button(self, window, button, act, mods):
        # update button state
        self.button_left_pressed = \
                glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        self.button_middle_pressed = \
                glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
        self.button_right_pressed = \
                glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS

        # update mouse position
        x, y = glfw.get_cursor_pos(window)
        self.last_mouse_x = int(self.scale * x)
        self.last_mouse_y = int(self.scale * y)

        if not self.model:
            return

        self.gui_lock.acquire()

        # save info
        if act == glfw.PRESS:
            self.last_button = button
            self.last_click_time = glfw.get_time()

        self.gui_lock.release()

    def handle_scroll(self, window, x_offset, y_offset):
        # require model
        if not self.model:
            return

        # get current window size
        width, height = glfw.get_framebuffer_size(window)

        # scroll
        self.gui_lock.acquire()
        mjlib.mjv_moveCamera(MOUSE_ZOOM, 0, (-20*y_offset), byref(self.cam), width, height)
        self.gui_lock.release()

    def should_stop(self):
        return glfw.window_should_close(self.window)

    def loop_once(self):
        self.render()
        # Swap front and back buffers
        glfw.swap_buffers(self.window)
        # Poll for and process events
        glfw.poll_events()

    def finish(self):
        glfw.terminate()
        mjlib.mjr_freeContext(byref(self.con))
        mjlib.mjv_freeObjects(byref(self.objects))
        self.running = False
