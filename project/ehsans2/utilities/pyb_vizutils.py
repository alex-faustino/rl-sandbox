import pybullet as p

class pyb_viz():
    def __init__(self, physicsClientId=0):
        self.physicsClientId = physicsClientId
        self.memorize_state()
        
    def memorize_state(self):
        self.body_unqids = []
        self.link_ids = []
        self.viz_data = []
        self.rgba_vals = []
        for i in range(p.getNumBodies(physicsClientId = self.physicsClientId)):
            body_unqid = p.getBodyUniqueId(i, physicsClientId = self.physicsClientId)
            viz_data = p.getVisualShapeData(body_unqid, physicsClientId = self.physicsClientId)
            for viz_lnk in viz_data:
                viz_lnk = viz_lnk[:]
                self.body_unqids.append(body_unqid)
                self.link_ids.append(viz_lnk[1])
                self.rgba_vals.append(viz_lnk[7])
                self.viz_data.append(viz_lnk)
    
    def find_idx(self, body_unqid, link_id):
        for i in range(len(self.body_unqids)):
            if self.body_unqids[i] == body_unqid and self.link_ids[i] == link_id:
                return i
        return None
        

    def unhide_all(self):
        for i in range(len(self.rgba_vals)):
            new_rgba = list(self.rgba_vals[i])[:]
            new_rgba[-1] = 1.
            p.changeVisualShape(objectUniqueId = self.body_unqids[i],
                                linkIndex = self.link_ids[i],
                                rgbaColor = new_rgba,
                                physicsClientId = self.physicsClientId)
    
    def hide_like_before(self):
        for i in range(len(self.rgba_vals)):
            new_rgba = list(self.rgba_vals[i])[:]
            p.changeVisualShape(objectUniqueId = self.body_unqids[i],
                                linkIndex = self.link_ids[i],
                                rgbaColor = new_rgba,
                                physicsClientId = self.physicsClientId)