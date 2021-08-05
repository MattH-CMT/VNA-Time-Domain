     
   
#This class computes a VNA time domain response using the Inverse Chirp Z transform
#https://www.nature.com/articles/s41598-019-50234-9/ "Generalizing the IFFT Off the Unit Circle" by Sukhoy and Stoytchev
#https://github.com/garrettj403/CZT Chirp Z library used
#Written by Matt Huebner

import numpy as np
import czt
class VNA_TDR:
    
    def __init__(self):
        self.unit = "seconds"#The units for the time domain axis("seconds", "meters", "feet")
        self.refType = 1#1 for round trip, #2 for two way
        self.VF = 1#Velocity factor from 0-1
        self.Beta = 0#Kaiser beta function
        self.Step = 0#Whether the impulse is being used for the step response or not
            
    def bandpass(self, X, Fstart, Fstop, Tstart, Tstop):
        """
        This function performs the inverse chirp z transform on the provided linear frequency data with bandpass windowing.
        The bandpass data does not need to be extended down to DC or have negative frequency points

         Parameters
         ----------
         X : Complex array
             S parameter frequency sweep in linear magnitude, generally more points is better.
        Fstart : double
            Start frequency for the frequency sweep.
        Fstop : double
             Stop Frequency for the frequency sweep.
        Tstart : double
            Start time for the time domain data.
        Tstop : double
            Stop time for the time domain data.
            
        Returns
        -------
        x0 : Array
            Time associated with each X1 point
            is changed to meters if Vf != 1
        X1 : Array
            Bandpass signal in linear scale with proper scaling

        """
        M = len(X)#Number of points of the frequency sweep
        freq = np.linspace(Fstart, Fstop, M)#Generate the linear frequency sweep of the VNA
        time = np.linspace(Tstart, Tstop, M)#Generate the linear time domain data for the frequency to be transformed onto
    
        freq = self.scaleFreq(freq)#VF Frequency Scaling
    
        #Windowing
        #A kaiser function is used to window the data. Beta corresponds to how much attenuation is applied to the frequency trace. 
        #A beta of zero corresponds to a rectangular window.
        #In the bandpass mode, the low and high frequencies will be attenuated the most from this function.
        #https://en.wikipedia.org/wiki/Kaiser_window
        kaiser = np.kaiser(M,self.Beta)
        scale_numerator = np.cumsum(kaiser)
        scale_denominator = np.cumsum(np.ones(M))
        scale = scale_numerator[M-1]/scale_denominator[M-1]#This scalar is used to compensate for the attenuation of the applied window
    
        windowedX = np.multiply(kaiser, X)#Multiplying by this window is essentially applying a band pass filter
        x = czt.freq2time(freq, windowedX, time)#Compute the time domain response and save to a tuple with the time sweep and magnitude
    
        x1 = abs(x[1])/scale#applying the scale to the time domain transform
        x0 = x[0]
        
        x0 = self.scaleTimeAxis(x0)#Scale off of units/VF
    
        return x0, x1#two arrays, first is time information and the second is the transform
    
    def lowpass_impulse(self, X, Fstart, Fstop, Tstart, Tstop):
        """
        This function performs the inverse chirp z transform on the provided linear frequency data with lowpass windowing
        This lowpass data does need to be extended down to DC with negative frequency points, but this does provide double 
        the resolution on the time axis.
         Parameters
         ----------
         X : Complex array
             S parameter frequency sweep in linear magnitude, generally more points is better.
        Fstart : double
            Start frequency for the frequency sweep.
        Fstop : double
            Stop Frequency for the frequency sweep.
        Tstart : double
            Start time for the time domain data.
        Tstop : double
            Stop time for the time domain data.

        Returns
        -------
        x0 : Array
            Time associated with each X1 point
            is changed to meters if Vf != 1
        X1 : Array
            Lowpass impulse in linear scale with proper scaling
        """
        M = len(X)#Number of points of the frequency sweep
    
        freq = np.linspace(Fstart, Fstop, M)#Generate the positive frequency sweep of the VNA
     
        negfreq = np.negative(np.flip(freq))#Reverse and multiply the previous frequency sweep by negative one
        freq = np.append(negfreq, np.append([0], freq))#Frequency sweep with negative, positive, and zero terms 
    
        freq = self.scaleFreq(freq)#VF frequency Scaling
    
        time = np.linspace(Tstart, Tstop, M)#Generate the linear time domain data for the frequency to be transformed onto
        zero = abs(X[0]) #DC Extrapolation is based off of the lowest frequency term, therefore the response is more accurate when you have more very low freq terms
        negX = np.flip(np.conj(X)) #Negative magnitudes are conjugate symetric. ie flipped and imaginary term multiplied by -1
        X = np.append(negX, np.append(zero, X))#Adding magnitude terms so there is DC and negative terms

    
        #Windowing
        #Now Applying a 2*M+1 sized window will  correspond to a low pass filter https://en.wikipedia.org/wiki/Kaiser_window
        kaiser =  np.kaiser(2*M+1, self.Beta)
        scale_numerator = np.cumsum(kaiser)
        scale_denominator = np.cumsum(np.ones(2*M+1))
        scale = scale_numerator[-1]/scale_denominator[-1]#This scalar is used to compensate for the attenuation of the applied window
    
        windowedX = np.multiply(kaiser, X)#applying the window to X
        x = czt.freq2time(freq, windowedX, time)#Computing the time domain response
 
        x0 = x[0]
        x1 = x[1]  
    
    
        if(self.Step == 0):#The scaling for attenuation does not matter if this function is being called for the lowpass step response
            x1 = abs(x[1])/scale#Accounting for attenuation from windowing, not applied to step response
        elif(self.Step == 1):
            self.toggleStep()
            
        x0 = self.scaleTimeAxis(x0)#Scale off of units/VF

        
        return x0,x1 #two arrays, first is time information and the second is the transform

    def lowpass_step(self, X, Fstart, Fstop, Tstart, Tstop):
        """
        This function accumulates the lowpass impulse function and scales the result to provide the linear lowpass step response.
        
        Interpolation could be used to force the data to be harmonically related
        Interpolation reference: https://stackoverflow.com/questions/19117660/how-to-generate-equispaced-interpolating-values 

        Parameters
        ----------
        X : Complex array
            S parameter frequency sweep in linear magnitude, generally more points is better.
        Fstart : double
            Start frequency for the frequency sweep.
        Fstop : double
            Stop Frequency for the frequency sweep.
        Tstart : double
            Start time for the time domain data.
        Tstop : double
            Stop time for the time domain data.

        Returns
        -------
        x0 : Array
            Time associated with each X1 point
            is changed to meters if Vf != 1
        X1 : Array
            Impulse step in linear scale with proper scaling

        """
        self.toggleStep()
        lowpass = self.lowpass_impulse(X, Fstart, Fstop, Tstart, Tstop)#Get the impulse response
    
        scale = (2*len(X)+1)/len(X)#Scaling occurs due to the increased resolution in the original plot that doesn't convey additional data
        x0 = lowpass[0]
        x1 = abs(np.cumsum(lowpass[1]))/scale#The step response is the accumulation(DT integration) of the impulse 
    
    
        return x0, x1#two arrays, first is time information and the second is the transform
    
    def scaleFreq(self, F):#Divides the frequency sweep by the VF, which corresponds to a time expansion, if unit is seconds don't do anything
        if(self.unit == "seconds"):
            return F
        else:
            return F/self.VF

    def scaleTimeAxis(self, T):
        if(self.unit == "seconds"):
            return T/self.refType#Scale by dividign by 2 or 1 depending on reflection type
        elif(self.unit == "feet"):
            return T*9.8357e8/self.refType#Scale by speed of light in feet/s and divide by 2 or 1 depending on reflection type
        elif(self.unit == "meters"):
            return T*2.9979e8/self.refType#Scale by speed of light in meters/s and divide by 2 or 1 depending on reflection type 
        else:
            return T
        

    def setUnit(self, str):#Sets the time domain response axis, default is in seconds
        x = str
        x = x.lower()
        if x == "seconds":
            unit = "seconds"
        elif x == "feet":
            unit = "feet"
        elif x == "meters":
            unit ="meters"
        else:
            unit = "seconds" 
        self.unit = unit
        
    def setVF(self, velfac):#Sets the Velocity factor of the response, if vf is negative or greater than 1, default to 1
        if((abs(velfac) <= 1.0) & (velfac > 0.0)):
            self.VF = velfac
        else:
            self.VF = 1
            
    def setBeta(self, B):#Sets the Kaiser window Beta, if beta is negative default to zero
        if(B >= 0):
            self.Beta = B
        else:
            self.Beta = 0
    
    def setRefType(self, reflection):#Sets the reflection type to either round trip(1) or one way(2), will default to one way if incorrect input
        if(reflection == 1):
            self.refType = reflection
        elif(reflection == 2):
            self.refType = reflection
        else:
            self.refType = 1
 
def toggleStep(self):#Toggles the scaling for both the step and impulse lowpass response, this function is just used internally
        if(self.Step == 1):
            self.Step = 0
        elif(self.Step == 0):
            self.Step = 1
