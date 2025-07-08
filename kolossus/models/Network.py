
import torch
from torch.utils.data import DataLoader
from torch import nn


class Cosine(nn.Module):
    def forward(self, kinases, site):
        return nn.CosineSimilarity()(kinases, site)

# Create Model
class Network(nn.Module):
    def __init__(self):
        super(Network,self).__init__()
        self.kinases_projector1 = nn.Sequential(nn.Linear(5120, 2048), nn.ReLU())
        nn.init.xavier_normal_(self.kinases_projector1[0].weight)
        
        self.kinases_projector2 = nn.Sequential(nn.Linear(2048, 1024), nn.ReLU())
        nn.init.xavier_normal_(self.kinases_projector2[0].weight)

        self.site_projector1 = nn.Sequential(nn.Linear(5120, 2048), nn.ReLU())
        nn.init.xavier_normal_(self.site_projector1[0].weight)
        
        self.site_projector2 = nn.Sequential(nn.Linear(2048, 1024), nn.ReLU())
        nn.init.xavier_normal_(self.site_projector2[0].weight)
        
        self.activator = Cosine()
        
    def forward(self, embedd):
          kinases = embedd[:, :5120]
          site = embedd[:, 5120:]  
          return self.classify(kinases, site)

    def classify(self, kinases, site):
        # get kinases and site projection
        site_projection, kinases_projection = self.project(kinases, site)

        # get cosine similarity (order doesnt' matter since cosine 
        # similarity is symmetric)
        distance = self.activator(kinases_projection, site_projection)

        return distance.squeeze(), site_projection, kinases_projection

    def project(self, kinases, site):
        # this is actually the site projection
        kinases_projection = self.kinases_projector1(kinases)
        kinases_projection = self.kinases_projector2(kinases_projection)

        # this is actually the kinase projection
        site_projection = self.site_projector1(site)
        site_projection = self.site_projector2(site_projection)

        # will actually return (projection of kinase, projection of site)
        return site_projection, kinases_projection


# for smaller version of ESM 
class NetworkSmall(nn.Module):
    def __init__(self):
        super(NetworkSmall, self).__init__()
        self.kinases_projector1 = nn.Sequential(nn.Linear(1280, 1024), nn.ReLU())
        nn.init.xavier_normal_(self.kinases_projector1[0].weight)
        
        self.kinases_projector2 = nn.Sequential(nn.Linear(1024, 612), nn.ReLU())
        nn.init.xavier_normal_(self.kinases_projector2[0].weight)

        self.site_projector1 = nn.Sequential(nn.Linear(1280, 1024), nn.ReLU())
        nn.init.xavier_normal_(self.site_projector1[0].weight)
        
        self.site_projector2 = nn.Sequential(nn.Linear(1024, 612), nn.ReLU())
        nn.init.xavier_normal_(self.site_projector2[0].weight)
        
        self.activator = Cosine()
        
    def forward(self, embedd):
          kinases = embedd[:, :1280]
          site = embedd[:, 1280:]  
          return self.classify(kinases, site)

    def classify(self, kinases, site):
        # get kinases and site projection
        site_projection, kinases_projection = self.project(kinases, site)

        # get cosine similarity (order doesnt' matter since cosine 
        # similarity is symmetric)
        distance = self.activator(kinases_projection, site_projection)

        return distance.squeeze(), site_projection, kinases_projection

    def project(self, kinases, site):
        # this is actually the site projection
        kinases_projection = self.kinases_projector1(kinases)
        kinases_projection = self.kinases_projector2(kinases_projection)

        # this is actually the kinase projection
        site_projection = self.site_projector1(site)
        site_projection = self.site_projector2(site_projection)

        # will actually return (projection of kinase, projection of site)
        return site_projection, kinases_projection
