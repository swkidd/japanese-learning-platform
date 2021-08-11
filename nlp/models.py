from django.db import models

class Word(models.Model):
    word = models.CharField(max_length=30)
    hira = models.CharField(max_length=30)
    known = models.FloatField()
    ex_sent = models.TextField()
    def __str__(self):
        return self.word