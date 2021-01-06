      PROGRAM climatology

c     *************************************************************************
c     * Compile cyclone climatology, extract cyclone fields                   *
c     * Michael Sprenger, Spring 2014                                         *
c     *************************************************************************

      use netcdf

      implicit none

c     ------------------------------------------------------------------------
c     Declaration of parameters
c     ------------------------------------------------------------------------

c     Dimension of ERAi domain
      real    xmin,ymin
      real    dx,dy
      integer nx,ny
      parameter (xmin=-180., ymin=-90.)
      parameter (dx=1.,dy=1.)
      parameter (nx=361,ny=181)

c     Time resolution of the cyclone climatology
      real      deltat
      parameter (deltat=6.)

c     Base directory for cyclone climatology
      character*200 bdir
      parameter
     >(bdir=
     >'/atmosdyn/erainterim/clim/front/4.0K_3ms_500km/cdf.filter/')

c     Numerical epsilon
      real eps
      parameter (eps = 0.0001)

c     ------------------------------------------------------------------------
c     Declaration of variables
c     ------------------------------------------------------------------------

c     Parameters
      integer      first(5)
      integer      final(5)
      character*80 outfile
      character*80 mode
      character*80 datemode,datefile
      character*80 outmode
      character*80 fieldname

c     netCDF variables
      character*100 cdfname
      character*80 varname
      integer      varid
      integer      cdfid
      integer      ierr
      real         misdat
      real         flag(nx,ny),fvel(nx,ny),the(nx,ny)
      real         out(nx,ny)
      real         count(nx,ny)

c     Single variable mode
      real          out_mean(nx,ny)
      real          out_count(nx,ny)
      real          out_sum(nx,ny)
      real          out_var(nx,ny)
      real          n(nx,ny),mean(nx,ny),M2(nx,ny),delta(nx,ny)

c     Auxiliary variables
      integer      i,j
      integer      date(5)
      integer      tmpdate(5)
      character*20 datestr
      integer      year,month,day,hour
      logical      file_exists
      integer      isok
      character*80 field(99),source(99),server(99),modelist(99),typ(99)
      character    ch
      integer      ileft,iright,ifound,imode,nmode,nfield
      character*80 str(10),tmpstr
      real         f_2d(nx,ny)

c     -----------------------------------------------------------------
c     Setup tables for raw data
c     -----------------------------------------------------------------

c     Setup table for data retrieval
      field ( 1)='fronts'
      source( 1)='L'
      server( 1)='free'
      typ   ( 1)='2d'

      field ( 2)='fronts-the'
      source( 2)='L'
      server( 2)='free'
      typ   ( 2)='2d'

      field ( 3)='fronts-fvel'
      source( 3)='L'
      server( 3)='free'
      typ   ( 3)='2d'

      nfield = 3

c     List of valid modes
      modelist( 1) = 'all'
      modelist( 2) = '1';   modelist( 3) = '01'; modelist( 4) = 'jan'
      modelist( 5) = '2';   modelist( 6) = '02'; modelist( 7) = 'feb'
      modelist( 8) = '3';   modelist( 9) = '03'; modelist(10) = 'mar'
      modelist(11) = '4';   modelist(12) = '04'; modelist(13) = 'apr'
      modelist(14) = '5';   modelist(15) = '05'; modelist(16) = 'may'
      modelist(17) = '6';   modelist(18) = '06'; modelist(19) = 'jun'
      modelist(20) = '7';   modelist(21) = '07'; modelist(22) = 'jul'
      modelist(23) = '8';   modelist(24) = '08'; modelist(25) = 'aug'
      modelist(26) = '9';   modelist(27) = '09'; modelist(28) = 'sep'
      modelist(29) = '10';  modelist(30) = 'oct'
      modelist(31) = '11';  modelist(32) = 'nov'
      modelist(33) = '12';  modelist(34) = 'dec'
      modelist(35) = 'djf'; modelist(36) = 'winter'
      modelist(37) = 'mam'; modelist(38) = 'spring'
      modelist(39) = 'jja'; modelist(40) = 'summer'
      modelist(41) = 'son'; modelist(42) = 'autumn'

      nmode = 41


c     ------------------------------------------------------------------------
c     Get parameters
c     ------------------------------------------------------------------------

c     Read parameters
      open(10,file='fort.10')
        read(10,*) datemode,datefile
        read(10,*) first(1),first(2),first(3),first(4)
        read(10,*) final(1),final(2),final(3),final(4)
        read(10,*) mode
        read(10,*) outfile
      close(10)

c     Init climatology arrays
      do i=1,nx
        do j=1,ny
          out(i,j)   = 0.
          count(i,j) = 0.
        enddo
      enddo

c     Init single field, level mode
      do i=1,nx
        do j=1,ny
          out_mean  (i,j)  = 0.
          out_sum   (i,j)  = 0.
          out_count (i,j)  = 0.
          out_var   (i,j)  = 0.
        enddo
      enddo

c     Init arrays for iterative variance calculation
      do i=1,nx
        do j=1,ny
            n(i,j)    = 0.
            mean(i,j) = 0.
            M2(i,j)   = 0.
        enddo
      enddo

c     Decompose the mode descriptor into different sections (. = separator)
      ifound = -1
      ileft  =  1
      iright = 80
      imode  =  0
      do i=1,80
        ch = mode(i:i)
        if ( ch.eq.'.' ) then
            iright     = i-1
            imode      = imode + 1
            str(imode) = mode(ileft:iright)
            ileft      = i + 1
        endif
      enddo
      imode      = imode + 1
      str(imode) = mode(ileft:80)

      print*
      print*,'------------ Parsing ------------------------------------'
      print*
      print*,trim(mode)
      print*
      do i=1,imode
         print*,i,trim(str(i))
      enddo
      print*

c     Get name of variable
      fieldname = 'climatology'
      do i=1,imode
        if ( str(i).eq.'fronts' ) then
           str(i)    = 'nil'
           fieldname = 'fronts'
         elseif ( str(i).eq.'fronts-the' ) then
           str(i)    = 'nil'
           fieldname = 'fronts-the'
         elseif ( str(i).eq.'fronts-fvel' ) then
           str(i)    = 'nil'
           fieldname = 'fronts-fvel'
        endif
      enddo

      print*,'------------ Field --------------------------------------'
      print*
      print*,'field : ',trim(fieldname)
      print*

c     Find the output mode
      outmode='mean'
      if ( fieldname.ne.'climatology' ) then
        do i=1,nmode
          do j=1,imode
             if ( str(j).eq.'dump' ) then
                outmode = 'dump'
                str(j)  = 'nil'
             endif
          enddo
        enddo
      endif

c     Find the mode
      do i=1,nmode
        do j=1,imode
           if ( str(j).eq.modelist(i) ) then
              mode   = modelist(i)
              str(j) = 'nil'
              goto 70
           endif
        enddo
      enddo

      mode = 'all'

 70   continue

      print*,'------------ Mode --------------------------------------'
      print*
      print*,'mode    : ',trim(mode)
      print*,'outmode : ',trim(outmode)
      print*

c     Check that all mode specifiers are handled
      do i=1,imode
        if ( str(i).ne.'nil' ) then
            print*,trim(str(i)),' not recognized... Stop'
            stop
        endif
      enddo

c     ------------------------------------------------------------------------
c     Compile the climatology
c     ------------------------------------------------------------------------

c     Loop over all dates
      if ( datemode.eq.'datestring' ) then
          do i=1,5
             date(i) = first(i)
          enddo
      elseif ( datemode.eq.'datelist' ) then
          open(20,file=datefile)
          read(20,*) datestr
          read(datestr(1:4  ),*) date(1)
          read(datestr(5:6  ),*) date(2)
          read(datestr(7:8  ),*) date(3)
          read(datestr(10:11),*) date(4)
      endif

100   continue

c     Decide whether to include file into climatology
      year  = date(1)
      month = date(2)
      day   = date(3)
      hour  = date(4)

      isok = 0

      if ( mode.eq.'all' ) isok = 1

      if ( mode.eq.'jan' .or. mode.eq.'1' .or. mode.eq.'01' ) then
        if ( month.ne.1 ) goto 120
        isok = 1
      endif
      if ( mode.eq.'feb' .or. mode.eq.'2' .or. mode.eq.'02' ) then
        if ( month.ne.2 ) goto 120
        isok = 1
      endif
      if ( mode.eq.'mar' .or. mode.eq.'3' .or. mode.eq.'03' ) then
        if ( month.ne.3 ) goto 120
        isok = 1
      endif
      if ( mode.eq.'apr' .or. mode.eq.'4' .or. mode.eq.'04' ) then
        if ( month.ne.4 ) goto 120
        isok = 1
      endif
      if ( mode.eq.'may' .or. mode.eq.'5' .or. mode.eq.'05' ) then
        if ( month.ne.5 ) goto 120
        isok = 1
      endif
      if ( mode.eq.'jun' .or. mode.eq.'6' .or. mode.eq.'06' ) then
        if ( month.ne.6 ) goto 120
        isok = 1
      endif
      if ( mode.eq.'jul' .or. mode.eq.'7' .or. mode.eq.'07' ) then
        if ( month.ne.7 ) goto 120
        isok = 1
      endif
      if ( mode.eq.'aug' .or. mode.eq.'8' .or. mode.eq.'08' ) then
        if ( month.ne.8 ) goto 120
        isok = 1
      endif
      if ( mode.eq.'sep' .or. mode.eq.'9' .or. mode.eq.'09' ) then
        if ( month.ne.9 ) goto 120
        isok = 1
      endif
      if ( mode.eq.'oct' .or. mode.eq.'10' ) then
        if ( month.ne.10 ) goto 120
        isok = 1
      endif
      if ( mode.eq.'nov' .or. mode.eq.'11' ) then
        if ( month.ne.11 ) goto 120
        isok = 1
      endif
      if ( mode.eq.'dec' .or. mode.eq.'12' ) then
        if ( month.ne.12 ) goto 120
        isok = 1
      endif

      if ( mode.eq.'djf' .or. mode.eq.'winter' ) then
        if ( month.ne.12 .and. month.ne.1 .and. month.ne.2 ) goto 120
        isok = 1
      endif
      if ( mode.eq.'mam' .or. mode.eq.'spring' ) then
        if ( month.ne.3 .and. month.ne.4 .and. month.ne.5 ) goto 120
        isok = 1
      endif
      if ( mode.eq.'jja' .or. mode.eq.'summer' ) then
        if ( month.ne.6 .and. month.ne.7 .and. month.ne.8 ) goto 120
        isok = 1
      endif
      if ( mode.eq.'son' .or. mode.eq.'autumn' ) then
        if ( month.ne.9 .and. month.ne.10 .and. month.ne.11 ) goto 120
        isok = 1
      endif

      if ( isok.eq.0 ) then
         print*,'Mode unregonized... stop'
         stop
      endif

      write(*,'(i5,i3,i3,i3)') (date(i),i=1,4)

c     Set the filename
      cdfname=bdir
      call datestring(datestr,date(1),-1,-1,-1)
      cdfname=trim(cdfname) // '/' // trim(datestr)
      call datestring(datestr,-1,date(2),-1,-1)
      cdfname=trim(cdfname) // '/' // trim(datestr)
      call datestring(datestr,date(1),date(2),date(3),date(4))
      cdfname=trim(cdfname) // '/L' // trim(datestr) // '.cdf'

c     Check whether file exists
      inquire(file=cdfname, exist=file_exists)
      if ( file_exists.eq..false. ) then
         print*,' WARNING : file is mising ',trim(cdfname)
         goto 120
      endif

c     Load the FLAG from climatology
      ierr = NF90_OPEN(TRIM(cdfname),nf90_nowrite, cdfid)
      IF ( ierr /= nf90_NoErr ) PRINT *,NF90_STRERROR(ierr)

      varname = 'LINE'
      ierr = NF90_INQ_VARID(cdfid,varname,varid)
      IF ( ierr /= nf90_NoErr) PRINT *,NF90_STRERROR(ierr)
      ierr = nf90_get_var(cdfid,varid,flag )
      IF ( ierr /= nf90_NoErr ) PRINT *,NF90_STRERROR(ierr)
      ierr = nf90_get_att(cdfid, varid, "_FillValue", misdat)
      IF ( ierr /= nf90_NoErr ) PRINT *,NF90_STRERROR(ierr)

      if ( fieldname.eq.'fronts-the') then
        varname = 'THE'
        ierr = NF90_INQ_VARID(cdfid,varname,varid)
        IF ( ierr /= nf90_NoErr) PRINT *,NF90_STRERROR(ierr)
        ierr = nf90_get_var(cdfid,varid,the )
        IF ( ierr /= nf90_NoErr ) PRINT *,NF90_STRERROR(ierr)
      endif

      if ( fieldname.eq.'fronts-the') then
        varname = 'FVEL'
        ierr = NF90_INQ_VARID(cdfid,varname,varid)
        IF ( ierr /= nf90_NoErr) PRINT *,NF90_STRERROR(ierr)
        ierr = nf90_get_var(cdfid,varid,fvel )
        IF ( ierr /= nf90_NoErr ) PRINT *,NF90_STRERROR(ierr)
      endif

      ierr = NF90_CLOSE(cdfid)
      IF( ierr /= nf90_NoErr) PRINT *,NF90_STRERROR(ierr)

c     Update climatology arrays
      if ( fieldname.eq.'climatology' ) then
        do i=1,nx
          do j=1,ny

            if ( abs(flag(i,j)-misdat).gt.eps ) then
               if ( flag(i,j).ne.0. ) out(i,j) = out(i,j) + 1.
               count(i,j)   = count(i,j) + 1.
            endif

           enddo
        enddo

c     Update single fields
      else

c       Copy the correct field to f_2d
        do i=1,nx
         do j=1,ny

            if ( fieldname.eq.'fronts' ) then
              if ( abs(flag(i,j)).gt.eps ) then
                  f_2d(i,j) = 1.
              else
                  f_2d(i,j) = 0.
              endif
            endif

            if ( fieldname.eq.'fronts-the' ) then
              if ( abs(flag(i,j)).gt.eps ) then
                  f_2d(i,j) = the(i,j)
              else
                  f_2d(i,j) = misdat
              endif
            endif

            if ( fieldname.eq.'fronts-fvel' ) then
              if ( abs(flag(i,j)).gt.eps ) then
                  f_2d(i,j) = fvel(i,j)
              else
                  f_2d(i,j) = misdat
              endif
            endif

          enddo
        enddo

c       count and sum
        do i=1,nx
          do j=1,ny

            if ( abs(f_2d(i,j)-misdat).gt.eps ) then
               out_sum(i,j)   = out_sum(i,j) + f_2d(i,j)
               out_count(i,j) = out_count(i,j) + 1.
            endif

           enddo
        enddo

c       online calculation of variance
        do i=1,nx
          do j=1,ny

            if ( abs(f_2d(i,j)-misdat).gt.eps ) then
             n(i,j)     = n(i,j) + 1
             delta(i,j) = f_2d(i,j) - mean(i,j)
             mean(i,j)  = mean(i,j) + delta(i,j) / n(i,j)
             M2(i,j)    = M2(i,j) + delta(i,j) * (f_2d(i,j) - mean(i,j))
            endif

           enddo
        enddo

      endif

c     Dump cdf files on request
      if ((outmode.eq.'dump').and.(fieldname.ne.'climatology')) then
         call datestring(datestr,date(1),date(2),date(3),date(4))
         cdfname='D'//trim(datestr)
         varname = trim(fieldname)
         call writecdf2D(cdfname,varname,flag,0.,
     >                dx,dy,xmin,ymin,nx,ny,1,1)
      endif

c     Go to next date
120   continue

      if ( datemode.eq.'datestring' ) then
         call newdate(date,deltat,tmpdate)
         do i=1,5
           date(i) = tmpdate(i)
         enddo
         if ( ( date(1).ne.final(1) ).or.
     >        ( date(2).ne.final(2) ).or.
     >        ( date(3).ne.final(3) ).or.
     >        ( date(4).ne.final(4) ) ) goto 100
      else
          read(20,*,end=130) datestr
          read(datestr(1:4  ),*) date(1)
          read(datestr(5:6  ),*) date(2)
          read(datestr(7:8  ),*) date(3)
          read(datestr(10:11),*) date(4)
          goto 100
130       continue
          close(20)
      endif


c     ------------------------------------------------------------------------
c     Write output to file
c     ------------------------------------------------------------------------

c     Normalize output
      if ( fieldname.eq.'climatology' ) then
        do i=1,nx
          do j=1,ny

           if ( count(i,j).gt.0. ) then
              out(i,j) = out(i,j) / count(i,j)
           else
              out(i,j) = misdat
           endif

          enddo
        enddo

c     Normalize output for single variable
      else

c       Get MEAN
        do i=1,nx
          do j=1,ny

             if ( out_count(i,j).gt.0. ) then
                out_mean(i,j) = out_sum(i,j) / out_count(i,j)
             else
                out_mean(i,j) = misdat
             endif

          enddo
        enddo

c       Set final variance
        do i=1,nx
          do j=1,ny
             if ( abs(n(i,j)-2.).lt.eps ) then
                out_var(i,j) = 0.
             else
                out_var(i,j) = M2(i,j) / ( n(i,j) - 1.)
             endif
          enddo
        enddo

      endif

c     Write climatology field
      if ( fieldname.eq.'climatology' ) then
        cdfname = outfile
        varname = 'FRONT.'//trim(mode)
        call writecdf2D(outfile,varname,out,0.,
     >                  dx,dy,xmin,ymin,nx,ny,1,1)

c     Write output for single level
      else
        cdfname = outfile
        varname = trim(fieldname)//'.MEAN'
        call writecdf2D(outfile,varname,out_mean,0.,
     >                  dx,dy,xmin,ymin,nx,ny,1,1)

        cdfname = outfile
        varname = trim(fieldname)//'.COUNT'
        call writecdf2D(outfile,varname,out_count,0.,
     >                  dx,dy,xmin,ymin,nx,ny,0,1)

        cdfname = outfile
        varname = trim(fieldname)//'.SUM'
        call writecdf2D(outfile,varname,out_sum,0.,
     >                  dx,dy,xmin,ymin,nx,ny,0,1)

        cdfname = outfile
        varname = trim(fieldname)//'.VAR'
        call writecdf2D(outfile,varname,out_var,0.,
     >                  dx,dy,xmin,ymin,nx,ny,0,1)

      endif

      end

c     --------------------------------------------------------------------
c     Subroutines to write the netcdf output file
c     --------------------------------------------------------------------

      subroutine writecdf2D(cdfname,
     >                      varname,arr,time,
     >                      dx,dy,xmin,ymin,nx,ny,
     >                      crefile,crevar)

c     Create and write to the netcdf file <cdfname>. The variable
c     with name <varname> and with time <time> is written. The data
c     are in the two-dimensional array <arr>. The list <dx,dy,xmin,
c     ymin,nx,ny> specifies the output grid. The flags <crefile> and
c     <crevar> determine whether the file and/or the variable should
c     be created

      IMPLICIT NONE

c     Declaration of input parameters
      character*80 cdfname,varname
      integer nx,ny
      real arr(nx,ny)
      real dx,dy,xmin,ymin
      real time
      integer crefile,crevar

c     Further variables
      real varmin(4),varmax(4),stag(4)
      integer ierr,cdfid,ndim,vardim(4)
      character*80 cstname
      real mdv
      integer datar(14),stdate(5)
      integer i

C     Definitions
      varmin(1)=xmin
      varmin(2)=ymin
      varmin(3)=1050.
      varmax(1)=xmin+real(nx-1)*dx
      varmax(2)=ymin+real(ny-1)*dy
      varmax(3)=1050.
      cstname=trim(cdfname)//'_cst'
      ndim=4
      vardim(1)=nx
      vardim(2)=ny
      vardim(3)=1
      stag(1)=0.
      stag(2)=0.
      stag(3)=0.
      mdv=-999.98999

C     Create the file
      if (crefile.eq.0) then
         call cdfwopn(cdfname,cdfid,ierr)
         if (ierr.ne.0) goto 906
      else if (crefile.eq.1) then
         call crecdf(cdfname,cdfid,
     >        varmin,varmax,ndim,cstname,ierr)
         if (ierr.ne.0) goto 903

C        Write the constants file
         datar(1)=vardim(1)
         datar(2)=vardim(2)
         datar(3)=int(1000.*varmax(2))
         datar(4)=int(1000.*varmin(1))
         datar(5)=int(1000.*varmin(2))
         datar(6)=int(1000.*varmax(1))
         datar(7)=int(1000.*dx)
         datar(8)=int(1000.*dy)
         datar(9)=1
         datar(10)=1
         datar(11)=0            ! data version
         datar(12)=0            ! cstfile version
         datar(13)=0            ! longitude of pole
         datar(14)=90000        ! latitude of pole
         do i=1,5
            stdate(i)=0
         enddo
c         call wricst(cstname,datar,0.,0.,0.,0.,stdate)
      endif

c     Write the data to the netcdf file, and close the file
      if (crevar.eq.1) then
         call putdef(cdfid,varname,ndim,mdv,
     >        vardim,varmin,varmax,stag,ierr)
         if (ierr.ne.0) goto 904
      endif
      call putdat(cdfid,varname,time,0,arr,ierr)
      if (ierr.ne.0) goto 905
      call clscdf(cdfid,ierr)

      return

c     Error handling
 903  print*,'*** Problem to create netcdf file ***'
      stop
 904  print*,'*** Problem to write definition ***'
      stop
 905  print*,'*** Problem to write data ***'
      stop
 906  print*,'*** Problem to open netcdf file ***'
      stop

      END

c     ---------------------------------------------------------------------------
c     Concatenate a date string
c     ---------------------------------------------------------------------------

      subroutine datestring(datestr,yyyy,mm,dd,hh)

c     Declaration of subroutine parameters
      integer yyyy,mm,dd,hh
      character*20 datestr

c     Auxiliary parameters
      integer f1,f2,i0
      integer yy,ce

      i0=ichar('0')
      datestr=''

      yy=mod(yyyy,100)
      ce=int(yyyy/100)

      if ((ce.ne.19).and.(ce.ne.20).and.(yyyy.gt.0)) then
         print*,'Invalid year... Stop'
         stop
      endif

      if (yy.ge.0) then
         f1=yy/10
         f2=mod(yy,10)
         if (ce.eq.19) then
            datestr=trim(datestr)//'19'//char(f1+i0)//char(f2+i0)
         else if (ce.eq.20) then
            datestr=trim(datestr)//'20'//char(f1+i0)//char(f2+i0)
         endif
      endif
      if (mm.gt.0) then
         f1=mm/10
         f2=mod(mm,10)
         datestr=trim(datestr)//char(f1+i0)//char(f2+i0)
      endif

      if (dd.gt.0) then
         f1=dd/10
         f2=mod(dd,10)
         datestr=trim(datestr)//char(f1+i0)//char(f2+i0)
      endif

      if (hh.ge.0) then
         f1=hh/10
         f2=mod(hh,10)
         datestr=trim(datestr)//'_'//char(f1+i0)//char(f2+i0)
      endif

      return
      end

c     ---------------------------------------------------------------------------
c     Calculates the new date when diff (in hours) is added to date1.
c     ---------------------------------------------------------------------------

      subroutine newdate(date1,diff,date2)
C
C     date1     int     input   array contains a date in the form
C                               year,month,day,hour,step
C     diff      real    input   timestep in hours to go from date1
C     date2     int     output  array contains new date in the same form

      integer   date1(5),date2(5)
      integer   idays(12)       ! array containing the days of the monthes
      real      diff
      logical   yearchange

      data idays/31,28,31,30,31,30,31,31,30,31,30,31/

      yearchange=.false.

      if ((mod(date1(1),4).eq.0).and.(date1(2).le.2)) idays(2)=29

      date2(1)=date1(1)
      date2(2)=date1(2)
      date2(3)=date1(3)
      date2(4)=date1(4)
      date2(5)=0
      date2(4)=date1(4)+int(diff)+date1(5)

      if (date2(4).ge.24) then
        date2(3)=date2(3)+int(date2(4)/24)
        date2(4)=date2(4)-int(date2(4)/24)*24
      endif
      if (date2(4).lt.0) then
        if (mod(date2(4),24).eq.0) then
          date2(3)=date2(3)-int(abs(date2(4))/24)
          date2(4)=date2(4)+int(abs(date2(4))/24)*24
        else
          date2(3)=date2(3)-(1+int(abs(date2(4))/24))
          date2(4)=date2(4)+(1+int(abs(date2(4))/24))*24
        endif
      endif

  100 if (date2(3).gt.idays(date2(2))) then
        if ((date2(2).eq.2).and.(mod(date2(1),4).eq.0)) idays(2)=29
        date2(3)=date2(3)-idays(date2(2))
        if (idays(2).eq.29) idays(2)=28
        date2(2)=date2(2)+1
        if (date2(2).gt.12) then
*         date2(1)=date2(1)+int(date2(2)/12)
*         date2(2)=date2(2)-int(date2(2)/12)*12
          date2(1)=date2(1)+1
          date2(2)=date2(2)-12
        endif
        if (date2(2).lt.1) then
          date2(1)=date2(1)-(1+int(abs(date2(2)/12)))
          date2(2)=date2(2)+(1+int(abs(date2(2)/12)))*12
        endif
        goto 100
      endif
  200 if (date2(3).lt.1) then
        date2(2)=date2(2)-1
        if (date2(2).gt.12) then
          date2(1)=date2(1)+int(date2(2)/12)
          date2(2)=date2(2)-int(date2(2)/12)*12
        endif
        if (date2(2).lt.1) then
          date2(1)=date2(1)-(1+int(abs(date2(2)/12)))
          date2(2)=date2(2)+(1+int(abs(date2(2)/12)))*12
        endif
        if ((date2(2).eq.2).and.(mod(date2(1),4).eq.0)) idays(2)=29
        date2(3)=date2(3)+idays(date2(2))
        if (idays(2).eq.29) idays(2)=28
        goto 200
      endif

      if (date2(2).gt.12) then
        date2(1)=date2(1)+int(date2(2)/12)
        date2(2)=date2(2)-int(date2(2)/12)*12
      endif
      if (date2(2).lt.1) then
        date2(1)=date2(1)-(1+int(abs(date2(2)/12)))
        date2(2)=date2(2)+(1+int(abs(date2(2)/12)))*12
      endif

      if (date2(1).lt.1000) then
      if (date2(1).ge.100) date2(1)=date2(1)-100
      endif

      return
      end



