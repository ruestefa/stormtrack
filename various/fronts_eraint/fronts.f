c     ---------------------------------------------------------------
      program fronts
c     ---------------------------------------------------------------

c     Atmospheric Dynamics Group IAC, ETH Zurich

c     Computation of location of fronts on THE charts
c     Advanced algorithm for front locating function used
c     Output with frontal strength

c     For advection speed: Output with negative sign for warm fronts 
c     and information of velocity and direction of motion

c     Optional output for frontal areas and removal of small objects
c     Optional removal of quasi-stationary fronts by min advection speeds

c     Literature:
c     -----------
c     Code is partly based on ideas of:
c     Hewson TD. 1998. Objective fronts. Met. Apps 5: 37–65, doi:10.1017/ S1350482798000553.

c     First development and application of our tool to COSMO Reanalysis:   
c     Jenkner J. et al. 2010. Detection and climatology of fronts in a high-resolution model reanalysis over the alps. Met. Apps 17: 1–18, doi:10.1002/met.142.

c     Introduction of frontal areas, mobile fronts, and first global climatology by our tool:
c     Schemm S. et al. 2014. Extratropical Fronts in the Lower Troposphere – Global Perspectives obtained from two Automated Methods. Quart. J. Roy. Meteor. Soc. 

c     History:
c     ---------
c     Johannes Jenkner April 2006
c                      September 2006
c                      September 2008
c     Sebastian Schemm October 2013
c     		           June 2014	

c     Development of front2line tool
c     Michael Sprenger April 2014

      implicit none

c     ---------------------------------------------------------------
c     Declaration of variables
c     ---------------------------------------------------------------

      character(len=100) :: innam,outnam
      character(len=80)  :: invar,cfn

      integer :: iargc,ierr,grid
      integer :: idin,idout,idcst
      integer :: ndim,ntimes,nmin
      integer :: i,j,k,n,nz,nlev,ntot

      integer, dimension(4) :: vardim
      integer, dimension(5) :: stdate

      real :: thld
      real :: misdat
      real :: pollon,pollat
      real :: dx,dy
      real :: phx,phy
      real :: minadv

      real,dimension(    4) :: varmin,varmax,stag
      real,dimension(10000) :: time

      real,dimension(:),allocatable :: aklev,bklev,aklay,bklay

      real,dimension(:,:),allocatable :: clusterin

      real,dimension(:,:,:),allocatable :: gradx,grady,vert,help,
     >                                     help1,help2,condi1,
     >                                     condi2,tfield,farea

c     MSP
      integer,dimension(:,:),allocatable :: belowc,abovec
      integer                               nbelow,nabove
      integer                               belowl(5000)
      integer                               il,ir,ju,jd
      integer                               ind,ival

      integer,dimension(:,:),allocatable :: clusterout,onecluster

      real,parameter :: eps=1.e-6

      logical :: winds,plus,withgap,area,mobile,objects
     
c      common /pole/ pollon,pollat     

c     -----------------------------------------------
c     Check number of input arguments
c     -----------------------------------------------

      print*,""
      print*," FRONT DETECTION "
      print*,""

      if ( (iargc().lt.4).or.(iargc().gt.10) ) then
        write(*,*) 'USAGE: ./fronts [-winds][-plus][-withgap]'//
     >   '[-area][-mobile thld][-objects minpixel]'//
     >	 ' INFILE OUTFILE VARIABLE THRESHOLD '

   	  print*," "
	  write(*,*) 'Optional parameters:'
	  write(*,*) ' -winds: output of advection velocity and direction'
	  write(*,*) ' -plus: enlarge the frontal lines'
	  write(*,*) ' -withgap: gaps between mobile sections'
	  write(*,*) ' -area: output frontal areas'
	  write(*,*) ' -mobile threshold: remove quasi-stationary fronts'//
     >		     ' by minimum advection criterion in m/s'
	  write(*,*) ' -objects minpixel: remove small objects with size'//
     >  	     ' lower or equal of "minpixel" grid points'

	  print*, " "
	  stop
    
      end if

c     -----------------------------------------------
c     Read input parameters and input file
c     -----------------------------------------------


      i=1
      call getarg(i,invar)
      winds=(invar=='-winds')
      if (winds) i=i+1

      call getarg(i,invar) 
      plus=(invar=='-plus')
      if (plus) i=i+1

      call getarg(i,invar)
      withgap=(invar=='-withgap')
      if (withgap) i=i+1

      call getarg(i,invar)
      area=(invar=='-area')
      if (area) i=i+1

      call getarg(i,invar)
      mobile=(invar=='-mobile')
      if (mobile) then
       i=i+1	
        call getarg(i,invar)
        read(invar,*) minadv
        i=i+1
      endif

      call getarg(i,invar)
      objects=(invar=='-objects')
      if (objects) then
       i=i+1	
        call getarg(i,invar)
        read(invar,*) nmin
        i=i+1
      endif
      
      call getarg(i,innam)
      call getarg(i+1,outnam)
      call getarg(i+3,invar)
      read(invar,*) thld
      call getarg(i+2,invar)


c     -----------------------------------------------
c     Print Info
c     -----------------------------------------------


      if (winds)   write(*,"(A)") "-winds option selected"
      if (plus)    write(*,"(A)") "-plus option selected"
      if (withgap) write(*,"(A)") "-withgap option selected"
      if (area)    write(*,"(A)") "-area option selected"
      if (mobile) write(*,"(A,F4.1)") "-mobile option selected: ",minadv
      if (objects) write(*,"(A,I4.1)") "-objects option selected:",nmin

c     -----------------------------------------------
c     Access to input netcdf file
c     -----------------------------------------------

      call cdfopn(trim(innam),idin,ierr)
      call getcfn(idin,cfn,ierr)
      call getdef(idin,trim(invar), 
     >            ndim,misdat,vardim,varmin,varmax,stag,ierr)    
      nz=vardim(3) 

c     -----------------------------------------------
c     Allocate required arrays
c     -----------------------------------------------

      allocate(help(vardim(1),vardim(2),nz))
      allocate(gradx(vardim(1),vardim(2),nz))
      allocate(grady(vardim(1),vardim(2),nz))
      allocate(help1(vardim(1),vardim(2),nz))
      allocate(help2(vardim(1),vardim(2),nz))
      allocate(vert(vardim(1),vardim(2),nz))
      allocate(condi1(vardim(1),vardim(2),nz))
      allocate(condi2(vardim(1),vardim(2),nz))
      allocate(tfield(vardim(1),vardim(2),nz))
      allocate(farea(vardim(1),vardim(2),nz))
      allocate(clusterout(vardim(1),vardim(2)))
      allocate(clusterin(vardim(1),vardim(2)))
      allocate(onecluster(vardim(1),vardim(2)))

c     MSP
      allocate(belowc(vardim(1),vardim(2)))
      allocate(abovec(vardim(1),vardim(2)))

c     -----------------------------------------------
c     Read in definitions of input netcdf file
c     -----------------------------------------------

      call cdfopn(cfn,idcst,ierr)  

c     Get grid dx / dy 
      call getgrid(idcst,dx,dy,ierr)
      call clscdf(idcst,ierr)           

c     Fill level reference array
      do i=1,nz
        vert(:,:,i)=real(i)
      enddo

c     Determine the x topology of the grid
c     2: periodic, cyclic point; 
c     1: periodic, not cyclic; 
c     0: not periodic (and therefore not closed)

c     dx   = (xmax-xmin) / real(nx-1)

      if (abs(varmax(1)-varmin(2)-360.).lt.eps) then
         grid=2
      elseif (abs(varmax(1)-varmin(2)-360.+dx).lt.eps) then
         grid=1
      else
         grid=0
      endif

c     Get time axis
      call gettimes(idin,time,ntimes,ierr) 

c     -----------------------------------------------      
c     Create output netcdf file
c     -----------------------------------------------

      call crecdf(trim(outnam),idout,varmin,varmax,3,
     >            trim(cfn),ierr)  

c     Define output variables
      call putdef(idout,'fronts_'//trim(invar),ndim,misdat,
     >            vardim,varmin,varmax,stag,ierr) 

      if (winds) then        
        call putdef(idout,'fvel_'//trim(invar),ndim,
     >              misdat,vardim,varmin,varmax,stag,ierr) 
        call putdef(idout,'fdir_'//trim(invar),ndim,
     >              misdat,vardim,varmin,varmax,stag,ierr) 
      endif
      
      if (area) then
        call putdef(idout,'farea_'//trim(invar),ndim,
     >              misdat,vardim,varmin,varmax,stag,ierr) 
      endif
      
      if (mobile) then
        call putdef(idout,'fmobile_'//trim(invar),ndim,
     >              misdat,vardim,varmin,varmax,stag,ierr)
      endif

      call putdef(idout,invar,ndim,
     >              misdat,vardim,varmin,varmax,stag,ierr) 

c     -----------------------------------------------
c     Loop over times
c     -----------------------------------------------

      do n=1,ntimes
        
        call getdat(idin,trim(invar),time(n),0,help,ierr)
        tfield = help
   
c     -----------------------------------------------     
c       Calculation of gradients and derived fields  
c     -----------------------------------------------

c	    1) Gradient of THE  

        call grad(gradx,grady,help,vert,varmin(1),varmin(2),
     >            dx,dy,vardim(1),vardim(2),nz,misdat)

c       2) Abs. value of gradient of THE (condi1)

        call abs2d(gradx,grady,condi1,vardim(1),vardim(2),
     >            nz,misdat)

c       3) Gradient of abs.value of gradient of THE (help1/help2)

        call grad(help1,help2,condi1,vert,varmin(1),varmin(2),
     >            dx,dy,vardim(1),vardim(2),nz,misdat)

c 	    4) Scalar product of 3) and 1) -> Gradient in direction of 1)
 
        call scal2d(help1,help2,gradx,grady,condi2,vardim(1),
     >            vardim(2),nz,misdat)

c       The TFP is condi2, not normalized by abs. value of 1)

c     -------------------------------------------------------
c       Calculate advection speed in direction of grad(TFP)
c     -------------------------------------------------------

        if (winds) then

          where (abs(condi2-misdat).gt.eps) condi2=-condi2
          
c        1) Gradient of TFP
          call grad(gradx,grady,condi2,vert,varmin(1),varmin(2),
     >            dx,dy,vardim(1),vardim(2),nz,misdat)
     
c        2) Normalized by abs. value of 1)
          call norm2d(gradx,grady,vardim(1),vardim(2),nz,misdat)

        endif  
   
c       --------------------------------------------------------------  
c       Calculation of masking condition: abs(grad TFP).gt. threshold 
c       --------------------------------------------------------------  

        where ((abs(condi1-misdat).gt.eps).and.(condi1.gt.thld))
          help=1
        elsewhere
          help=0
        endwhere  
        
c       --------------------------------------------------------------  
c       Save this area as frontal area
c       --------------------------------------------------------------  

        where ((abs(condi1-misdat).gt.eps).and.(condi1.gt.thld))
          farea=1
        elsewhere
          farea=0
        endwhere 	
        
c       --------------------------------------------------------------  
c       Calculation of locating condition; zero TFP
c       --------------------------------------------------------------  

        call zcontour(condi2,vardim(1),vardim(2),nz,
     >                varmin(1),varmin(2),dx,dy,misdat)   
       
        help = help*condi2    
     
c       --------------------------------------------------------------       
c       Avoid detection of minimum gradient
c       --------------------------------------------------------------  

        call div(condi2,help1,help2,vert,varmin(1),
     >           varmin(2),dx,dy,vardim(1),vardim(2),nz,misdat)


        where (condi2.gt.0) help=0         
        where (condi2.gt.0) farea=0
 
c       -----------------------------------------------    
c       Apply special emphasizing filter for lines
c       -----------------------------------------------
        if (plus) then
        
         call linesplus(help,vardim(1),vardim(2),nz,
     >                   varmin(1),varmin(2),dx,dy,misdat)
        endif  

c       --------------------------------------------------------------  
c       Scale output with frontal intensity
c       --------------------------------------------------------------  

        help  = help *condi1
        farea = farea*condi1
        
c       --------------------------------------------------------------         
c       Set vacant pixels to missing
c       --------------------------------------------------------------  

        where (help ==0)  help = misdat
        where (farea==0) farea = misdat

c       --------------------------------------------------------------  
c       Further specification of frontal characteristics by winds  
c       --------------------------------------------------------------  

        if (winds.or.mobile) then

          call getdat(idin,'U',time(n),0,help1,ierr)
          call getdat(idin,'V',time(n),0,help2,ierr)

c       --------------------------------------------------------------  
c       Advection speed in direction of normalized TFP
c       --------------------------------------------------------------  

          call scal2d(help1,help2,gradx,grady,condi1,vardim(1),
     >              vardim(2),nz,misdat) 

c         Rotate wind vector if necessary (COSMO only)
c          if (abs(pollat-90.).gt.eps) then
c            do k=1,nz
c              do j=1,vardim(2)
c                phy=varmin(2)+(j-1)*dy 
c                do i=1,vardim(1)
c                  phx=varmin(1)+(i-1)*dx
c                  call t_phuv2lluv(gradx(i,j,k),grady(i,j,k),
c     >                             help1(i,j,k),help2(i,j,k),phx,phy)
c                enddo
c              enddo
c            enddo
c          else
            help1=gradx
            help2=grady
c          endif 
   
c       --------------------------------------------------------------        
c        Correction of frontal direction for warm fronts
c       --------------------------------------------------------------  

          where ((abs(condi1-misdat).gt.eps).and.(condi1.lt.0.))
            help1 = -help1
            help2 = -help2
          end where

          call winddir(vardim(1),vardim(2),nz,misdat,help1,help2,
     >                 condi2)

          where (abs(help-misdat).lt.eps) 
            condi1 = misdat
            condi2 = misdat
          end where

c       --------------------------------------------------------------  
c       end further specifications in terms of winds
c       --------------------------------------------------------------  

      endif

c     --------------------------------------------------------------  
c     Remove quasi-stationary fronts 
c     --------------------------------------------------------------  
     
      if (mobile.and.(withgap.eq..false.)) then

c       MSP BEGIN
        do k=1,vardim(3)

c        Cluster all stationary frontal sections
         do i=1,vardim(1)
           do j=1,vardim(2)

              if ( ( abs(condi1(i,j,k))       .lt.minadv ).and.
     >             ( abs(condi1(i,j,k)-misdat).gt.eps    ).and.
     >             ( abs(condi2(i,j,k)-misdat).gt.eps    ) )
     >        then
                 clusterin(i,j) = 1
              else
                 clusterin(i,j) = 0
              endif
           enddo
         enddo
         call clustering(belowc,nbelow,clusterin,
     >                   vardim(1),vardim(2),grid)

c        Cluster all mobile frontal sections
         do i=1,vardim(1)
           do j=1,vardim(2)

              if ( ( abs(condi1(i,j,k))       .ge.minadv ).and.
     >             ( abs(condi1(i,j,k)-misdat).gt.eps    ).and.
     >             ( abs(condi2(i,j,k)-misdat).gt.eps    ) )
     >        then
                 clusterin(i,j) = 1
              else
                 clusterin(i,j) = 0
              endif
           enddo
         enddo
         call clustering(abovec,nabove,clusterin,
     >                   vardim(1),vardim(2),grid)

c       Init the removal flag for each stationary section
        do j=1,nbelow
          belowl(i) = 0
        enddo

c       Check wether the stationary clusters are connected to at least two
c       different mobilr clusters -> belowl=-1
        do i=1,vardim(1)
          do j=1,vardim(2)

             ind = belowc(i,j)
             if ( ind.ne.0 ) then
               il = i-1
               ir = i+1
               ju = j+1
               jd = j-1
               if (il.lt.1         ) il = 1
               if (ir.gt.vardim(1) ) ir = vardim(1)
               if (jd.lt.1         ) jd = 1
               if (ju.gt.vardim(2) ) ju = vardim(2)

               ival = abovec(il,j )
               if ( (ival.ne.0).and.(belowl(ind).eq.0) ) then
                  belowl(ind) = ival
               elseif ( (ival.ne.0).and.(belowl(ind).ne.ival) ) then
                  belowl(ind) = -1
               endif

               ival = abovec(ir,j )
               if ( (ival.ne.0).and.(belowl(ind).eq.0) ) then
                  belowl(ind) = ival
               elseif ( (ival.ne.0).and.(belowl(ind).ne.ival) ) then
                  belowl(ind) = -1
               endif

               ival = abovec(i ,jd)
               if ( (ival.ne.0).and.(belowl(ind).eq.0) ) then
                  belowl(ind) = ival
               elseif ( (ival.ne.0).and.(belowl(ind).ne.ival) ) then
                  belowl(ind) = -1
               endif

               ival = abovec(i ,ju)
               if ( (ival.ne.0).and.(belowl(ind).eq.0) ) then
                  belowl(ind) = ival
               elseif ( (ival.ne.0).and.(belowl(ind).ne.ival) ) then
                  belowl(ind) = -1
               endif

               ival = abovec(il,jd)
               if ( (ival.ne.0).and.(belowl(ind).eq.0) ) then
                  belowl(ind) = ival
               elseif ( (ival.ne.0).and.(belowl(ind).ne.ival) ) then
                  belowl(ind) = -1
               endif

               ival = abovec(ir,jd)
               if ( (ival.ne.0).and.(belowl(ind).eq.0) ) then
                  belowl(ind) = ival
               elseif ( (ival.ne.0).and.(belowl(ind).ne.ival) ) then
                  belowl(ind) = -1
               endif

               ival = abovec(il,ju)
               if ( (ival.ne.0).and.(belowl(ind).eq.0) ) then
                  belowl(ind) = ival
               elseif ( (ival.ne.0).and.(belowl(ind).ne.ival) ) then
                  belowl(ind) = -1
               endif

               ival = abovec(ir,ju)
               if ( (ival.ne.0).and.(belowl(ind).eq.0) ) then
                  belowl(ind) = ival
               elseif ( (ival.ne.0).and.(belowl(ind).ne.ival) ) then
                  belowl(ind) = -1
               endif

             endif

            enddo
          enddo

c         Remove all stationary sections which have not two mobile neighbors
          do i=1,vardim(1)
            do j=1,vardim(2)

               ind = belowc(i,j)
               if ( ind.gt.0 ) then
                 if ( belowl(ind).ne.-1 ) then
                    help(i,j,k)   = misdat
                    condi1(i,j,k) = misdat
                    condi2(i,j,k) = misdat
                 endif
               endif

            enddo
          enddo


        enddo

      elseif ( mobile.and.(withgap.eq..true.) ) then

        print*,'Applying advection criterion'
c 	    check if advection velocity is below threshold
        where(abs(condi1).lt.minadv)          condi1 = misdat
        where(abs(condi1-misdat).lt.eps)      condi2 = misdat
        where(abs(condi2-misdat).lt.eps)      help   = misdat

      endif


c     --------------------------------------------------------------  
c     Remove very short objects
c     --------------------------------------------------------------  


      if (objects) then	

       do k=1,vardim(3)

          where ((abs(help(:,:,k)-misdat).lt.eps))
            clusterin(:,:)=0.
          elsewhere
            clusterin(:,:)=1.
          endwhere

     	 call clustering(clusterout,ntot,clusterin,
     >			        vardim(1),vardim(2),grid)


          do j=1,ntot
          
            if (count(clusterout(:,:)==j).lt.nmin) then
               where (clusterout(:,:)==j) help(:,:,k) = misdat
            endif
                
          enddo  

       enddo

c     --------------------------------------------------------------  
c     correction to fvel_var/fdir_var after min objects removal
c     --------------------------------------------------------------  

       if (winds.or.mobile) then
       
        where(abs(help-misdat).lt.eps) condi1 = misdat
        where(abs(help-misdat).lt.eps) condi2 = misdat
        
       end if
       
c     --------------------------------------------------------------  
c      correction to frontal areas after min objects removal
c     --------------------------------------------------------------  

       if (area) then

        clusterout = 0

        do k=1,vardim(3)

          where ((abs(farea(:,:,k)-misdat).lt.eps))
            clusterin=0.
          elsewhere
            clusterin=1.
          endwhere

     	  call clustering(clusterout,ntot,clusterin,
     >                      vardim(1),vardim(2),grid)

        do  j = 1, ntot
            onecluster=0
            where (clusterout(:,:)==j) onecluster=clusterout

            if ( .not. any( (onecluster(:,:)*help(:,:,k)).gt.0) ) then
                where (clusterout(:,:)==j) farea(:,:,k)=misdat
            endif

          enddo 

        enddo
c     --------------------------------------------------------------  
c     end if area
c     --------------------------------------------------------------  

       endif

c     --------------------------------------------------------------  
c     end if objects
c     --------------------------------------------------------------  

      endif

c     --------------------------------------------------------------  
c      Output 
c     --------------------------------------------------------------  


        if (winds) then
       
          call putdat(idout,'fvel_'//trim(invar),time(n),0,condi1,
     >                ierr)    
          call putdat(idout,'fdir_'//trim(invar),time(n),0,condi2,
     >                ierr)    
        endif           

        if (area) then
          call putdat(idout,'farea_'//trim(invar),time(n),0,farea,
     >                ierr)   

        endif        

        call putdat(idout,'fronts_'//trim(invar),time(n),0,
     >               help,ierr)   

        call putdat(idout,invar,time(n),0,tfield,ierr)   

        if (mobile) then
           help = 0.
           where( abs(condi1).lt.minadv     ) help = 1.
           where( abs(condi1).ge.minadv     ) help = 2.
           where( abs(condi1-misdat).le.eps ) help = 0.

           call putdat(idout,'fmobile_'//trim(invar),time(n),0,
     >               help,ierr)

        endif

   
c     --------------------------------------------------------------  
      enddo ! time loop
c     --------------------------------------------------------------  

c     --------------------------------------------------------------  
c     Close netcdf files
c     --------------------------------------------------------------  

      call clscdf(idin,ierr)
      call clscdf(idout,ierr) 


c     --------------------------------------------------------------  
c     Deallocate required arrays
c     --------------------------------------------------------------  

       if (allocated(help))  deallocate(help)
       if (allocated(gradx)) deallocate(gradx)
       if (allocated(grady)) deallocate(grady)
       if (allocated(help1)) deallocate(help1)
       if (allocated(help2)) deallocate(help2)
       if (allocated(condi1)) deallocate(condi1)
       if (allocated(condi2)) deallocate(condi2)
       if (allocated(vert))   deallocate(vert)
       if (allocated(tfield)) deallocate(tfield)
       if (allocated(farea))  deallocate(farea)
       if (allocated(clusterout)) deallocate(clusterout)
       if (allocated(clusterin))  deallocate(clusterin)
       if (allocated(onecluster)) deallocate(onecluster)


        print*," "
        print*," DONE"
        print*," "

c     -----------------------------------------------
      end program fronts
c     -----------------------------------------------
